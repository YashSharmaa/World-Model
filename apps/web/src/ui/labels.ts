// ─────────────────────────────────────────────────────────────────────────────
//  GPU-Center World Model — three.js Decision Studio (React overlay)
//  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
//  License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────────────
//  Human labels for the kernel-safe candidate alphabet (hold / admit-a / admit-b)
//  derived in the store. Kept in one place so every panel names actions the same.

/** Full action label for the recommendation panel + workload card chips. */
export function actionLabel(actionId: string): string {
  if (actionId === "hold") return "Hold one window";
  if (actionId.startsWith("admit-a")) return "Admit → Rack Group A";
  if (actionId.startsWith("admit-b")) return "Admit → Rack Group B";
  return actionId;
}

/** Short branch label for the delta ribbon columns. */
export function branchLabel(actionId: string): string {
  if (actionId === "hold") return "Hold";
  if (actionId.startsWith("admit-a")) return "Admit A";
  if (actionId.startsWith("admit-b")) return "Admit B";
  return actionId;
}
