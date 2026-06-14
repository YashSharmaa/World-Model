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
  Group,
  Line,
  Mesh,
  PlaneGeometry,
  Vector3,
} from "three";
import type { RackDTO } from "../api/client";
import type { MaterialSet } from "./materials";
import type { RackView } from "./racks";

export interface NetworkResult {
  root: Group;
  switches: Map<string, Mesh>;
  links: Group;
}

/** Build ToR switch meshes (one per rack) + a small inter-rack link mesh set. */
export function buildNetwork(
  rackViews: RackView[],
  mats: MaterialSet,
): NetworkResult {
  const root = new Group();
  root.name = "network_root";
  const switches = new Map<string, Mesh>();
  const links = new Group();
  links.name = "inter_rack_link_lines";

  for (const rv of rackViews) {
    const sw = new Mesh(new BoxGeometry(1.4, 0.3, 0.5), mats.switchMesh);
    sw.position.set(rv.center.x, 4.6, rv.center.z);
    sw.name = `tor_switch_${rv.rackId}`;
    sw.userData = { kind: "node", id: `tor-${rv.rackId}` };
    switches.set(rv.rackId, sw);
    root.add(sw);
  }

  // A few high-value inter-rack links (spine fan-out), not a full mesh.
  for (let i = 0; i < rackViews.length; i++) {
    for (let j = i + 1; j < rackViews.length; j++) {
      // only adjacent-in-block + one cross-block link to keep it readable
      const sameBlock =
        rackViews[i].rackId.slice(0, 6) === rackViews[j].rackId.slice(0, 6);
      if (!sameBlock && (i + j) % 3 !== 0) continue;
      const a = new Vector3(rackViews[i].center.x, 4.6, rackViews[i].center.z);
      const b = new Vector3(rackViews[j].center.x, 4.6, rackViews[j].center.z);
      const geo = new BufferGeometry().setFromPoints([a, b]);
      const line = new Line(geo, mats.interRackLine);
      links.add(line);
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

  // topology domain hull: one translucent box per rack domain.
  for (const rv of rackViews) {
    const hull = new Mesh(new BoxGeometry(4.0, 5.0, 2.8), mats.domainHull);
    hull.position.set(rv.center.x, 2.5, rv.center.z);
    hull.name = `topology_hull_dom-${rv.rackId}`;
    hull.userData = { kind: "domain", id: `dom-${rv.rackId}` };
    topologyHulls.add(hull);
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
  return { root, topologyHulls, powerRails, coolingPlanes };
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
