// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//
//  network_root + domain_root (doc G scene graph).
//
//    network_root : a ToR switch mesh per rack + a few inter-rack link lines.
//                   No packet animation — topology QUALITY matters, packet-level
//                   detail adds noise and false precision (doc G deferred list).
//    domain_root  : translucent, READ-ONLY context layers — topology domain
//                   hulls, power-domain rails, cooling-zone planes. These are
//                   risk context, not control knobs (doc G assumptions).

import {
  BoxGeometry,
  BufferGeometry,
  Color,
  Group,
  Line,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  Vector3,
} from "three";
import type { RackDTO, RelationDTO } from "../api/client";
import { PALETTE, type MaterialSet } from "./materials";
import type { RackView } from "./racks";

export interface NetworkResult {
  root: Group;
  switches: Map<string, Mesh>;
  links: Group;
}

/**
 * Build ToR switch meshes (one per rack) + the inter-rack link mesh set.
 *
 * When a `LINKED_TO` switch→switch relation is present in `relations`, the link
 * geometry is built from THOSE pairs (the authoritative spine fabric the backend
 * declared) rather than an index heuristic — so the topology the demo shows is
 * the topology the snapshot actually carries (doc G: relations drive geometry).
 */
export function buildNetwork(
  rackViews: RackView[],
  mats: MaterialSet,
  relations: RelationDTO[] = [],
): NetworkResult {
  const root = new Group();
  root.name = "network_root";
  const switches = new Map<string, Mesh>();
  const links = new Group();
  links.name = "inter_rack_link_lines";

  // switch_id -> rack center, so a switch→switch relation can be drawn in space.
  const centerBySwitch = new Map<string, Vector3>();
  for (const rv of rackViews) {
    const sw = new Mesh(new BoxGeometry(1.4, 0.3, 0.5), mats.switchMesh);
    sw.position.set(rv.center.x, 4.6, rv.center.z);
    sw.name = `tor_switch_${rv.rackId}`;
    sw.userData = { kind: "node", id: `tor-${rv.rackId}` };
    switches.set(rv.rackId, sw);
    centerBySwitch.set(`tor-${rv.rackId}`, new Vector3(rv.center.x, 4.6, rv.center.z));
    root.add(sw);
  }

  const addLink = (a: Vector3, b: Vector3): void => {
    const geo = new BufferGeometry().setFromPoints([a, b]);
    links.add(new Line(geo, mats.interRackLine));
  };

  // Prefer the declared switch→switch spine fabric from the relation array.
  const spine = relations.find(
    (r) => r.src_kind === "switch" && r.dst_kind === "switch",
  );
  if (spine && spine.pairs.length) {
    for (const [srcSwitch, dstSwitch] of spine.pairs) {
      const a = centerBySwitch.get(srcSwitch);
      const b = centerBySwitch.get(dstSwitch);
      if (a && b) addLink(a, b);
    }
  } else {
    // Fallback (no spine relation): a few high-value links, not a full mesh.
    for (let i = 0; i < rackViews.length; i++) {
      for (let j = i + 1; j < rackViews.length; j++) {
        const sameBlock =
          rackViews[i].rackId.slice(0, 6) === rackViews[j].rackId.slice(0, 6);
        if (!sameBlock && (i + j) % 3 !== 0) continue;
        addLink(
          new Vector3(rackViews[i].center.x, 4.6, rackViews[i].center.z),
          new Vector3(rackViews[j].center.x, 4.6, rackViews[j].center.z),
        );
      }
    }
  }
  root.add(links);
  return { root, switches, links };
}

export interface DomainResult {
  root: Group;
  topologyHulls: Group;
  powerRails: Group;
  coolingPlanes: Group;
  /** domain_id -> its topology-hull mesh, for annotation emphasis. */
  hullByDomain: Map<string, Mesh>;
}

/** Build translucent topology / power / cooling context layers. */
export function buildDomains(
  racks: RackDTO[],
  rackViews: RackView[],
  mats: MaterialSet,
): DomainResult {
  const root = new Group();
  root.name = "domain_root";

  const topologyHulls = new Group();
  topologyHulls.name = "topology_domain_hulls";
  const powerRails = new Group();
  powerRails.name = "power_domain_rails";
  const coolingPlanes = new Group();
  coolingPlanes.name = "cooling_zone_planes";

  const viewById = new Map(rackViews.map((v) => [v.rackId, v]));
  const hullByDomain = new Map<string, Mesh>();

  // topology domain hull: one translucent box per rack domain. Each hull gets
  // its OWN material clone so annotation emphasis (blocked / rejected) can recolour
  // a single domain without affecting the others.
  for (const rv of rackViews) {
    const domainId = `dom-${rv.rackId}`;
    const hull = new Mesh(new BoxGeometry(4.0, 5.0, 2.8), mats.domainHull.clone());
    hull.position.set(rv.center.x, 2.5, rv.center.z);
    hull.name = `topology_hull_${domainId}`;
    hull.userData = { kind: "domain", id: domainId };
    topologyHulls.add(hull);
    hullByDomain.set(domainId, hull);
  }

  // power-domain rails: a thin translucent slab spanning racks sharing a PDU.
  const byPower = new Map<string, RackDTO[]>();
  for (const r of racks) {
    (byPower.get(r.power_domain_id) ?? byPower.set(r.power_domain_id, []).get(r.power_domain_id)!).push(r);
  }
  for (const [pdu, group] of byPower) {
    const centers = group
      .map((g) => viewById.get(g.rack_id))
      .filter((v): v is RackView => !!v)
      .map((v) => v.center);
    if (!centers.length) continue;
    const minX = Math.min(...centers.map((c) => c.x)) - 2;
    const maxX = Math.max(...centers.map((c) => c.x)) + 2;
    const z = centers[0].z;
    const rail = new Mesh(new BoxGeometry(maxX - minX, 0.1, 0.5), mats.powerRail);
    rail.position.set((minX + maxX) / 2, 0.05, z - 1.8);
    rail.name = `power_rail_${pdu}`;
    powerRails.add(rail);
  }

  // cooling-zone planes: a translucent floor patch per cooling zone.
  const byCooling = new Map<string, RackDTO[]>();
  for (const r of racks) {
    (byCooling.get(r.cooling_zone_id) ?? byCooling.set(r.cooling_zone_id, []).get(r.cooling_zone_id)!).push(r);
  }
  for (const [zone, group] of byCooling) {
    const centers = group
      .map((g) => viewById.get(g.rack_id))
      .filter((v): v is RackView => !!v)
      .map((v) => v.center);
    if (!centers.length) continue;
    const plane = new Mesh(new PlaneGeometry(6, 4), mats.coolingPlane);
    plane.rotation.x = -Math.PI / 2;
    const cx = centers.reduce((s, c) => s + c.x, 0) / centers.length;
    const cz = centers.reduce((s, c) => s + c.z, 0) / centers.length;
    plane.position.set(cx, 0.02, cz);
    plane.name = `cooling_plane_${zone}`;
    coolingPlanes.add(plane);
  }

  root.add(topologyHulls, powerRails, coolingPlanes);
  return { root, topologyHulls, powerRails, coolingPlanes, hullByDomain };
}

/** Default (idle) tint + opacity for a topology hull, restored each frame. */
const HULL_IDLE = { color: new Color(PALETTE.domainHull), opacity: 0.06 };

/**
 * Apply rollout / why-not scene annotations to the domain layer (doc G
 * scene_annotations + visual_states): blocked domains glow hard red with a stop
 * intent, rejected (why-not) domains glow amber, and the recommended/candidate
 * domain gets a soft safe pulse. Every hull is reset to idle first so stale
 * emphasis never lingers across frames.
 */
export function applyDomainAnnotations(
  domains: DomainResult,
  opts: {
    blockedDomainIds?: string[];
    rejectedDomainIds?: string[];
    recommendedDomainIds?: string[];
    pulse?: number; // 0..1 animation phase for the candidate-target pulse
  },
): void {
  const blocked = new Set(opts.blockedDomainIds ?? []);
  const rejected = new Set(opts.rejectedDomainIds ?? []);
  const recommended = new Set(opts.recommendedDomainIds ?? []);
  const pulse = 0.5 + 0.5 * Math.abs(Math.sin((opts.pulse ?? 0) * Math.PI * 2));

  for (const [domainId, hull] of domains.hullByDomain) {
    const mat = hull.material as MeshBasicMaterial;
    if (blocked.has(domainId)) {
      mat.color.copy(PALETTE.blocked);
      mat.opacity = 0.22;
    } else if (rejected.has(domainId)) {
      mat.color.copy(PALETTE.alt);
      mat.opacity = 0.18;
    } else if (recommended.has(domainId)) {
      // candidate-target subtle pulse on the feasible/recommended domain.
      mat.color.copy(PALETTE.safe);
      mat.opacity = 0.08 + 0.1 * pulse;
    } else {
      mat.color.copy(HULL_IDLE.color);
      mat.opacity = HULL_IDLE.opacity;
    }
    mat.needsUpdate = true;
  }
}

/** Toggle visibility of the three read-only context layers. */
export function setDomainLayerVisibility(
  domains: DomainResult,
  layer: "topology" | "power" | "cooling",
  visible: boolean,
): void {
  const g =
    layer === "topology"
      ? domains.topologyHulls
      : layer === "power"
        ? domains.powerRails
        : domains.coolingPlanes;
  g.visible = visible;
}
