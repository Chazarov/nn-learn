export type ProjectType = "PERCEPTRON" | "KOHONEN";

export type ProjectSummary = {
  id: string;
  project_type: ProjectType;
  user_id: string;
  created_at: number;
  csv_file_id: string;
};

export type NNData = {
  input_size: number;
  mins: number[];
  maxs: number[];
  classes: string[];
  weights: number[][][];
};

export type ProjectWithData = ProjectSummary & {
  nn_data: NNData;
};
