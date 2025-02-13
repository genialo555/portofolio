"use client";

import React, { useEffect, useRef } from "react";

interface Point {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface CanvasRevealEffectProps {
  colors?: number[][];
  dotSize?: number;
  opacities?: number[];
  animationSpeed?: number;
  containerClassName?: string;
}

export const CanvasRevealEffect = ({
  colors = [[255, 255, 255]],
  dotSize = 2,
  opacities = [1],
  animationSpeed = 5,
  containerClassName = "",
}: CanvasRevealEffectProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const points = useRef<Point[]>([]);
  const animationFrameId = useRef<number>();
  const mousePosition = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    };

    const initPoints = () => {
      const { width, height } = canvas.getBoundingClientRect();
      points.current = Array.from({ length: 50 }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * animationSpeed,
        vy: (Math.random() - 0.5) * animationSpeed,
      }));
    };

    const drawLines = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < points.current.length; i++) {
        const point = points.current[i];
        const colorIndex = i % colors.length;
        const opacityIndex = i % opacities.length;

        // Update position
        point.x += point.vx;
        point.y += point.vy;

        // Bounce off walls
        if (point.x < 0 || point.x > canvas.width) point.vx *= -1;
        if (point.y < 0 || point.y > canvas.height) point.vy *= -1;

        // Draw point
        ctx.beginPath();
        ctx.arc(point.x, point.y, dotSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${colors[colorIndex].join(",")},${opacities[opacityIndex]})`;
        ctx.fill();

        // Draw lines to nearby points
        for (let j = i + 1; j < points.current.length; j++) {
          const otherPoint = points.current[j];
          const dx = point.x - otherPoint.x;
          const dy = point.y - otherPoint.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(otherPoint.x, otherPoint.y);
            ctx.strokeStyle = `rgba(${colors[colorIndex].join(",")},${opacities[opacityIndex] * (1 - distance / 100)})`;
            ctx.stroke();
          }
        }
      }

      animationFrameId.current = requestAnimationFrame(drawLines);
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mousePosition.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    resizeCanvas();
    initPoints();
    drawLines();

    window.addEventListener("resize", resizeCanvas);
    canvas.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      canvas.removeEventListener("mousemove", handleMouseMove);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [colors, dotSize, opacities, animationSpeed]);

  return (
    <div className={containerClassName}>
      <canvas
        ref={canvasRef}
        className="h-full w-full"
        style={{ touchAction: "none" }}
      />
    </div>
  );
}; 