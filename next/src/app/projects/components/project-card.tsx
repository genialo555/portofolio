"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";

interface ProjectCardProps {
  title: string;
  description: string;
  image: string;
  demoUrl: string;
  features: string[];
  tags: string[];
}

export function ProjectCard({
  title,
  description,
  image,
  demoUrl,
  features,
  tags,
}: ProjectCardProps) {
  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl bg-white">
      <Image
        src={image}
        alt={title}
        fill
        className="object-cover object-center transition duration-200"
      />
    </div>
  );
}