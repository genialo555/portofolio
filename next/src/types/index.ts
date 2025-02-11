export interface Card {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  type: "project" | "widget";
  tags: string[];
  demoUrl: string;
  features: string[];
  className?: string;
  content?: React.ReactNode;
}
