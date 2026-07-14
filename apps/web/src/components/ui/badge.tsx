// shadcn/ui — Badge (extended with the app's status variants). License: MIT.
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold leading-none transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-secondary text-secondary-foreground",
        outline: "text-foreground",
        safe: "border-transparent bg-[var(--safe-bg)] text-safe",
        blocked: "border-transparent bg-[var(--blocked-bg)] text-blocked",
        abstain: "border-border bg-secondary text-lowconf",
        conf: "border-border bg-secondary text-muted-foreground tabular-nums",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
