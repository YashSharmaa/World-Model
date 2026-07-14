// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  Bootstrap. ONE WebGLRenderer, ONE PerspectiveCamera rig with fixed presets,
//  ONE world_root scene graph, ONE HTML overlay shell. The render loop is the
//  only place scene state is mutated per frame, and it is driven by the store's
//  shared time cursor + backend rollout points (interpolation only — doc G
//  "Timeline authority").
//
//  Everything is a SEMANTIC READ MODEL: the renderer reads typed entities,
//  relations, and rollout diffs and projects them onto instanced primitives.
//  No game-engine simulation runs in the browser.

import {
  Clock,
  Color,
  type LineBasicMaterial,
  type LineSegments,
  NoToneMapping,
  PCFSoftShadowMap,
  PerspectiveCamera,
  Scene,
  SRGBColorSpace,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";

import { apiClient } from "./api/client";
import { Store, type StoreState } from "./state/store";
import { buildSceneGraph, type SceneGraph } from "./scene/sceneGraph";
import { applyOccupancy, baselineOccupancyByRack } from "./scene/racks";
import {
  applyWorkloadHighlight,
  commitToken,
  setAllocationPath,
  updateCommitTweens,
} from "./scene/workloads";
import { applyDomainAnnotations, setDomainLayerVisibility } from "./scene/overlays";
import { rebuildBranches, updateBranches } from "./scene/branches";
import { CameraDirector } from "./routes/modes";
import { installEnvironment, type EnvironmentHandle } from "./scene/environment";
import { buildProAssets, type ProAssetsResult } from "./scene/proAssets";
import { createLoaderArm } from "./scene/loaderArm";
import { createPostFX } from "./render/postfx";
import { CinematicDirector, defaultGoldenPath } from "./anim/director";
import { getSceneTheme, onTheme, type SceneTheme } from "./theme";
import "./index.css";
import { mountOverlay } from "./ui/mount";
import type { DomainLayer, ReplayTrace, StudioBridge } from "./ui/studio-context";

/** Playback speed: scene-seconds advanced per real second when playing. */
const PLAYBACK_RATE = 120; // 2 min of horizon per real second (recordable pace)

async function main(): Promise<void> {
  const canvas = document.getElementById("webgl-canvas") as HTMLCanvasElement;
  const overlay = document.getElementById("dom-overlay") as HTMLElement;

  // ── renderer + camera ──────────────────────────────────────────────────
  // Pro look: postprocessing owns tone mapping (renderer = NoToneMapping),
  // SRGB output, PCFSoft shadows. In three 0.169 lights are physically correct
  // by default (the legacy-lights flag was removed), so nothing extra is needed.
  const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(new Color(getSceneTheme().background), 1);
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = NoToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;

  const scene = new Scene();
  const camera = new PerspectiveCamera(45, 1, 0.1, 400);
  const director = new CameraDirector(camera);
  const cinematic = new CinematicDirector(camera);

  // cinematic lighting + IBL + atmosphere + reflective floor (replaces the old
  // flat 3-light setup with a baked-environment 3-point rig).
  const environment: EnvironmentHandle = installEnvironment(scene, renderer);
  void environment; // retained for disposal; kept alive for the session.

  // ── post-processing pipeline (ACES tone map + bloom + SMAA + vignette) ────
  const postfx = createPostFX(renderer, scene, camera);

  function resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    postfx.setSize(w, h);
  }
  resize();
  window.addEventListener("resize", resize);

  // ── store + scene graph ─────────────────────────────────────────────────
  const store = new Store(apiClient);
  await store.boot();

  const initial = store.get().snapshot;
  if (!initial) throw new Error("no snapshot");
  let sceneGraph: SceneGraph = buildSceneGraph(initial);
  scene.add(sceneGraph.worldRoot);
  let lastStateHash = initial.state_hash;

  // ── professional set dressing (decorative; reads the rack positions) ──────
  // Built from the same rack layout the semantic builder produced. Rebuilt with
  // the scene graph on a state_hash change; never picked, never mutates
  // applyOccupancy / commit / branch state.
  let proAssets: ProAssetsResult = buildProAssets(sceneGraph.racks);
  sceneGraph.worldRoot.add(proAssets.root);

  // ── theme reactivity for the semantic scene graph ─────────────────────────
  // The environment + postfx subscribe to the theme themselves; here we retune
  // the scene graph's OWN floor/grid (built dark by default) so the inner pod
  // floor matches the bright studio floor in light mode (no dark square clash).
  function applyWorldTheme(t: SceneTheme): void {
    renderer.setClearColor(new Color(t.background), 1);
    const fm = sceneGraph.materials.floor;
    fm.color.set(t.floorColor);
    fm.metalness = t.floorMetalness;
    fm.roughness = t.floorRoughness;
    fm.needsUpdate = true;
    sceneGraph.worldRoot.traverse((o) => {
      if (o.type === "GridHelper") {
        const ls = o as LineSegments;
        const mat = (
          Array.isArray(ls.material) ? ls.material[0] : ls.material
        ) as LineBasicMaterial;
        mat.vertexColors = false;
        mat.color.set(t.gridColor);
        mat.transparent = true;
        mat.opacity = t.gridOpacity;
        mat.needsUpdate = true;
      }
    });
  }
  onTheme(applyWorldTheme); // fires immediately with the current theme

  // ── the rigged loader arm (SkinnedMesh + AnimationMixer) ──────────────────
  // Mounted at the queue-lane edge so its "place" motion sweeps into the pod.
  const arm = createLoaderArm();
  arm.object3D.position.set(-6.5, 0, 4.5);
  scene.add(arm.object3D);
  /** action_id we last dramatised, so we trigger a place exactly once per commit. */
  let lastPlacedActionId: string | null = null;

  // Author attribution travels with the persistent HTML footer credit
  // (bottom-right) and the per-payload `credit` field; an in-scene 3D credit
  // plate was removed because it projected over the centre of the pod and
  // clashed with the "Current pod state" label and the scene itself.

  // ── 2D overlay bridge (React owns the overlay; main owns the scene) ───────
  // The React overlay (shadcn/ui + Tailwind) reads the same store and writes
  // back through this bridge. Domain context-layer visibility is the scene's
  // source of truth here; the React Controls panel mirrors it and calls
  // setLayer() to mutate the live graph.
  const layerVisible: Record<DomainLayer, boolean> = {
    topology: true,
    power: true,
    cooling: true,
  };
  function setLayer(layer: DomainLayer, visible: boolean): void {
    layerVisible[layer] = visible;
    setDomainLayerVisibility(sceneGraph.domains, layer, visible);
  }
  /** Re-apply the operator's context-layer visibility after a geometry rebuild. */
  function reapplyLayerVisibility(): void {
    (Object.keys(layerVisible) as DomainLayer[]).forEach((layer) =>
      setDomainLayerVisibility(sceneGraph.domains, layer, layerVisible[layer]),
    );
  }

  // Replay trace (predicted / realized / both) the render loop overlays; the
  // React Timeline toggle writes it through setTrace().
  let replayTrace: ReplayTrace = "both";
  const setTrace = (t: ReplayTrace): void => {
    replayTrace = t;
  };

  // Self-captioning beat text signal: written by BOTH the cinematic demo and the
  // store-derived caption, read by the React <BeatCaption/> via useSyncExternalStore.
  const beatListeners = new Set<() => void>();
  let beatText: string | null = null;
  function setBeat(text: string | null): void {
    if (text === beatText) return;
    beatText = text;
    for (const l of beatListeners) l();
  }
  const beatSignal = {
    get: () => beatText,
    subscribe: (cb: () => void) => {
      beatListeners.add(cb);
      return () => beatListeners.delete(cb);
    },
  };

  // ── golden-path "Play demo" control + auto-start ──────────────────────────
  // The cinematic director runs the eight doc-G beats; each beat's onBegin
  // drives real store state (focus → compare → play A → play B → hold → freeze →
  // why-not → resolve). While the demo runs, its per-beat caption owns the
  // muted on-screen caption (LinkedIn autoplays muted), so we suppress the
  // store-driven caption to avoid the two fighting.
  let goldenPathActive = false;
  function playDemo(): void {
    goldenPathActive = true;
    store.setFocusedWorkload(
      store.get().focusedWorkload ??
        store.get().snapshot?.entities.workloads.find(
          (w) => w.status === "pending",
        )?.workload_uid ??
        null,
    );
    cinematic.playGoldenPath(
      defaultGoldenPath(store, (text) => {
        if (text) setBeat(text);
      }),
    );
  }

  // Mount the React 2D overlay (shadcn/ui + Tailwind) over the canvas. The
  // bridge hands React the store, the API client, and the imperative scene
  // callbacks main.ts owns (play demo, toggle a context layer, set replay trace,
  // the beat-caption signal).
  const bridge: StudioBridge = {
    store,
    api: apiClient,
    playDemo,
    setLayer,
    initialLayers: { ...layerVisible },
    setTrace,
    beat: beatSignal,
  };
  mountOverlay(overlay, bridge);

  // ── raycasting (interaction_root) ────────────────────────────────────────
  const ndc = new Vector2();
  function toNdc(ev: PointerEvent): Vector2 {
    ndc.set(
      (ev.clientX / window.innerWidth) * 2 - 1,
      -(ev.clientY / window.innerHeight) * 2 + 1,
    );
    return ndc;
  }
  canvas.addEventListener("pointerdown", (ev) => {
    // ignore clicks that started on an overlay panel
    if ((ev.target as HTMLElement) !== canvas) return;
    const sel = sceneGraph.pick(toNdc(ev), camera);
    store.setSelection(sel);
    if (sel) store.setMode("inspect-drawer");
  });

  // limited orbit drag (no free-fly): horizontal/vertical offset only.
  const orbit = { x: 0, y: 0 };
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
  });
  window.addEventListener("pointerup", () => (dragging = false));
  window.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    orbit.x = Math.max(-0.6, Math.min(0.6, orbit.x + (ev.clientX - lastX) * 0.003));
    orbit.y = Math.max(-0.5, Math.min(0.8, orbit.y - (ev.clientY - lastY) * 0.003));
    lastX = ev.clientX;
    lastY = ev.clientY;
  });

  // ── store subscription: structural updates ───────────────────────────────
  let currentState: StoreState = store.get();
  store.subscribe((s) => {
    currentState = s;
    // Rebuild geometry only when the authoritative state_hash changes (cheap
    // semantic-diff guard — we never rebuild per frame).
    if (s.snapshot && s.snapshot.state_hash !== lastStateHash) {
      scene.remove(sceneGraph.worldRoot);
      proAssets.dispose();
      sceneGraph.dispose();
      sceneGraph = buildSceneGraph(s.snapshot);
      proAssets = buildProAssets(sceneGraph.racks);
      sceneGraph.worldRoot.add(proAssets.root);
      scene.add(sceneGraph.worldRoot);
      applyWorldTheme(getSceneTheme()); // keep the fresh floor/grid on-theme
      reapplyLayerVisibility(); // preserve the operator's context-layer toggles
      lastStateHash = s.snapshot.state_hash;
    }
    // Rebuild branch ghosts when the compare result or pinned set changes.
    if (s.compare) {
      rebuildBranches(
        sceneGraph.branches,
        s.compare.candidates,
        s.pinnedActions,
        sceneGraph.racks.racks,
        sceneGraph.materials,
      );
    }
    director.goTo(s.mode);
    sceneGraph.highlight(s.selection);
    // While the golden-path demo runs, its per-beat caption owns the caption;
    // otherwise the store-driven caption tracks the live state. (The React
    // panels re-render themselves from the store via useSyncExternalStore.)
    if (!goldenPathActive) updateBeatCaption(s);
  });

  function updateBeatCaption(s: StoreState): void {
    // Self-captioning headline that tracks the demo's beats (doc G storyboard).
    if (s.mode === "replay-calibration" && s.replay) {
      setBeat(s.cursorS >= 720 ? "Predicted vs realized — divergence" : "Replay: predicted branch");
      return;
    }
    if (!s.compare) {
      setBeat(s.focusedWorkload ? "Current pod state · pick an action" : null);
      return;
    }
    if (s.whyNotActionId) {
      setBeat("Rejected for exact blockers and higher risk");
      return;
    }
    const rec = s.compare.candidates.find((c) => c.action_id === s.compare?.recommended_action_id);
    setBeat(rec ? `Compare futures — recommended: ${rec.action_id}` : "Compare futures");
  }

  // ── render loop ──────────────────────────────────────────────────────────
  // per-frame scratch maps for the thermal/activity dressing (rackId -> 0..1).
  const heatByRack: Record<string, number> = {};
  const utilByRack: Record<string, number> = {};

  const clock = new Clock();
  function frame(): void {
    // The whole frame body is guarded so a single transient error (e.g. a
    // malformed branch payload) can never kill the rAF loop and blank the scene;
    // the reschedule in `finally` keeps the render loop alive regardless.
    try {
      const dt = clock.getDelta();
      // Advance the shared time cursor (interpolation only, backend-authoritative).
      store.advance(dt * PLAYBACK_RATE);

      // Camera: the cinematic director may take over (golden-path / named shots);
      // when it yields, the mode CameraDirector drives the limited-orbit preset.
      if (!cinematic.update(dt)) {
        director.tick(orbit);
      }
      // When the golden-path beat sequence finishes, hand the caption back to the
      // live store-driven caption.
      if (goldenPathActive && !cinematic.isSequenceActive()) {
        goldenPathActive = false;
      }

      applyRolloutToScene(currentState);

      // Advance the fixed-duration workload commit tweens off wall-clock dt.
      updateCommitTweens(sceneGraph.workloads, dt);

      // Drive the rigged loader arm + the professional set dressing.
      arm.update(dt);
      proAssets.update(clock.elapsedTime, heatByRack, utilByRack);

      // Render through the post-processing composer (ACES + bloom + SMAA + vignette).
      postfx.render(dt);
    } catch (err) {
      console.error("[decision-studio] frame error (render loop kept alive):", err);
    } finally {
      requestAnimationFrame(frame);
    }
  }

  /**
   * Project the rollout point at the current shared cursor onto the scene:
   * grounded occupancy, branch ghosts, allocation paths, halos. This is the
   * heart of the read model — semantic diffs in, instanced primitives out.
   */
  function applyRolloutToScene(s: StoreState): void {
    const cursor = s.cursorS;
    const horizon = Math.max(1, s.horizonS);

    // Grounded baseline: the snapshot's current occupancy. The grounded pod is
    // ALWAYS held at this in decision-studio and compare modes — the predicted
    // future is layered as a translucent ghost ON TOP, never promoted to the
    // solid grounded fill (doc G core_views: grounded now vs ghost future).
    const baseline = s.snapshot
      ? baselineOccupancyByRack(sceneGraph.racks, s.snapshot.entities.gpus)
      : {};

    // In replay-calibration the scene is driven by the replay traces, not the
    // live snapshot baseline, so skip the grounded reset there.
    const replayMode = s.mode === "replay-calibration" && s.replay;

    if (!replayMode) {
      // 0) Ground EVERY rack at the snapshot baseline (solid fill).
      for (const view of sceneGraph.racks.racks) {
        const rackId = view.rackId;
        const occ = baseline[rackId] ?? 0;
        const hot = s.snapshot?.entities.racks.find((r) => r.rack_id === rackId)?.hotspot_score ?? 0;
        applyOccupancy(sceneGraph.racks, rackId, occ, hot, false, occ);
        heatByRack[rackId] = hot;
        utilByRack[rackId] = occ;
      }
    }

    // 1) Branch ghosts (compare / decision modes). The predicted occupancy is an
    // overlay drawn in the ghost colour ABOVE the grounded baseline.
    if (s.compare && s.pinnedActions.length && !replayMode) {
      const focus =
        s.mode === "decision-studio" ? s.compare.recommended_action_id : null;
      updateBranches(sceneGraph.branches, cursor, horizon, focus);

      // The fronted branch (recommended in studio, pinnedActions[0] in compare)
      // drives the GHOST overlay occupancy + path + commit so the pod visibly
      // *previews* filling as the cursor advances — without ever becoming solid.
      const recId =
        s.mode === "counterfactual-compare"
          ? s.pinnedActions[0]
          : s.compare.recommended_action_id;
      const rec = s.compare.candidates.find((c) => c.action_id === recId);
      const point = sampleRollout(rec?.rollout?.points ?? [], cursor);
      if (point) {
        for (const [rackId, occ] of Object.entries(point.rack_occupancy)) {
          const hot = point.rack_hotspot[rackId] ?? 0;
          const base = baseline[rackId] ?? 0;
          // ghost=true: slots beyond the grounded baseline read predicted/translucent.
          applyOccupancy(sceneGraph.racks, rackId, occ, hot, true, base);
          heatByRack[rackId] = hot;
          utilByRack[rackId] = occ;
        }
        // commit the focused workload token over the fixed commit window.
        const targetDomain = rec?.rollout?.scene_annotations.recommended_domain_ids[0];
        const rackId = targetDomain?.replace("dom-", "") ?? null;
        const rackView = rackId
          ? sceneGraph.racks.racks.find((r) => r.rackId === rackId) ?? null
          : null;
        if (s.focusedWorkload) {
          setAllocationPath(sceneGraph.workloads, s.focusedWorkload, rackView, sceneGraph.materials);
          // Trigger a fixed-duration wall-clock commit tween once the branch
          // begins committing (cursor past 0); the easing runs off frame dt.
          commitToken(
            sceneGraph.workloads,
            s.focusedWorkload,
            cursor > 0 ? rackView : null,
          );
        }

        // Dramatise the world-model "place" intervention: when the planner's
        // recommended branch is an admit_and_place toward a real rack, the
        // rigged loader arm sweeps into that rack ONCE per committed action.
        const isPlace = rec?.action_id != null && rackView != null;
        if (isPlace && rec!.action_id !== lastPlacedActionId && cursor > 0) {
          lastPlacedActionId = rec!.action_id;
          const target = rackView!.center.clone().add(new Vector3(0, 1.0, 0));
          arm.playPlace(target);
        }
      }

      // Apply scene_annotations to the domain + workload layers: blocked domains
      // (from blocked branches' rollouts), the recommended/candidate domain, the
      // why-not rejected domains, and the highlighted workloads (doc G).
      const blockedDomainIds: string[] = [];
      const highlightWorkloadIds: string[] = [];
      for (const cand of s.compare.candidates) {
        const ann = cand.rollout?.scene_annotations;
        if (!ann) continue;
        if (!cand.safe) blockedDomainIds.push(...ann.blocked_domain_ids);
        highlightWorkloadIds.push(...ann.highlight_workload_ids);
      }
      const recommendedDomainIds =
        rec?.rollout?.scene_annotations.recommended_domain_ids ?? [];
      // why-not rejected domains light amber when the why-not card is open.
      const rejectedDomainIds: string[] = [];
      if (s.whyNotActionId) {
        const wc = s.compareCandidates[s.whyNotActionId];
        const dom = (wc?.params.domain_id_or_set as string[] | undefined)?.[0];
        if (dom) rejectedDomainIds.push(dom);
      }
      const phase = (cursor / horizon) % 1;
      applyDomainAnnotations(sceneGraph.domains, {
        blockedDomainIds,
        rejectedDomainIds,
        recommendedDomainIds,
        pulse: phase,
      });
      applyWorkloadHighlight(
        sceneGraph.workloads,
        s.focusedWorkload
          ? [s.focusedWorkload, ...highlightWorkloadIds]
          : highlightWorkloadIds,
        phase,
      );
    } else if (!s.compare) {
      // Clear stale annotations when no comparison is active.
      applyDomainAnnotations(sceneGraph.domains, {});
      applyWorkloadHighlight(
        sceneGraph.workloads,
        s.focusedWorkload ? [s.focusedWorkload] : [],
        (cursor / horizon) % 1,
      );
      // No active comparison: let the arm settle back to its idle pose and clear
      // the last-placed guard so the next commit re-triggers the place motion.
      if (lastPlacedActionId !== null) {
        lastPlacedActionId = null;
        arm.toIdle();
      }
    }

    // 2) Replay overlay (predicted vs realized).
    if (replayMode && s.replay) {
      const trace = replayTrace;
      const predPoint = sampleRollout(s.replay.predicted.points, cursor);
      const realPoint = sampleRollout(s.replay.realized.points, cursor);

      if (trace === "both" && predPoint && realPoint) {
        // Render predicted (ghost) and realized (solid) SIMULTANEOUSLY: the
        // grounded base is the realized history, with the predicted EXTRA fill
        // layered translucent on top per rack (doc G replay divergence).
        for (const view of sceneGraph.racks.racks) {
          const rackId = view.rackId;
          const realOcc = realPoint.rack_occupancy[rackId] ?? 0;
          const predOcc = predPoint.rack_occupancy[rackId] ?? 0;
          const realHot = realPoint.rack_hotspot[rackId] ?? 0;
          // solid realized base:
          applyOccupancy(sceneGraph.racks, rackId, realOcc, realHot, false, realOcc);
          // predicted ghost overlay where prediction exceeds realized:
          if (predOcc > realOcc) {
            applyOccupancy(sceneGraph.racks, rackId, predOcc, realHot, true, realOcc);
          }
          heatByRack[rackId] = realHot;
          utilByRack[rackId] = realOcc;
        }
      } else {
        const usePred = trace === "predicted";
        const rollout = usePred ? s.replay.predicted : s.replay.realized;
        const point = sampleRollout(rollout.points, cursor);
        if (point) {
          for (const [rackId, occ] of Object.entries(point.rack_occupancy)) {
            const hot = point.rack_hotspot[rackId] ?? 0;
            applyOccupancy(sceneGraph.racks, rackId, occ, hot, usePred, usePred ? 0 : occ);
            heatByRack[rackId] = hot;
            utilByRack[rackId] = occ;
          }
        }
      }
    }
  }

  requestAnimationFrame(frame);

  // ── auto-start the golden-path demo ───────────────────────────────────────
  // Recordability (doc G): the studio is a self-running recorded explanation, so
  // it kicks off the eight-beat storyboard shortly after boot. The "Play demo"
  // control re-runs it on demand. Opt out with ?autostart=0 for manual driving.
  const autostart =
    new URLSearchParams(window.location.search).get("autostart") !== "0";
  if (autostart) {
    window.setTimeout(() => playDemo(), 1200);
  }

  // expose a tiny debug handle (handy when recording / scripting beats):
  //   __studio.playDemo()                  — run the 8-beat golden-path demo
  //   __studio.cinematic.cutTo("rackCloseup") — jump to a named shot
  //   __studio.cinematic.release()          — hand the camera back to mode presets
  //   __studio.arm.playPlace(new THREE.Vector3(...)) — replay the loader place
  (window as unknown as { __studio: unknown }).__studio = {
    store,
    sceneGraph,
    camera,
    cinematic,
    arm,
    proAssets,
    playDemo,
  };
}

/** Linear sample of a rollout's points at time t (the renderer interpolates). */
function sampleRollout<T extends { t_s: number }>(points: T[], t: number): T | null {
  if (!points.length) return null;
  let chosen = points[0];
  for (const p of points) {
    if (p.t_s <= t) chosen = p;
    else break;
  }
  return chosen;
}

main().catch((err) => {
  // Surface boot failures visibly rather than a blank canvas.
  const overlay = document.getElementById("dom-overlay");
  if (overlay) {
    const div = document.createElement("div");
    div.className = "panel";
    div.style.cssText = "left:50%;top:50%;transform:translate(-50%,-50%)";
    div.textContent = `Decision Studio failed to start: ${String(err)}`;
    overlay.appendChild(div);
  }
  console.error(err);
});
