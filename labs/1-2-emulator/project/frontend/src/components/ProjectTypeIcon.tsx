import type { ProjectType } from "../types";

const titles: Record<ProjectType, string> = {
  PERCEPTRON: "Многослойный перцептрон",
  KOHONEN: "Сеть Кохонена (SOM)",
};

export function ProjectTypeIcon({ type }: { type: ProjectType }) {
  const isP = type === "PERCEPTRON";
  return (
    <span
      className={`project-type-icon ${isP ? "perceptron" : "kohonen"}`}
      title={titles[type]}
      aria-hidden
    >
      {isP ? "⎔" : "▦"}
    </span>
  );
}
