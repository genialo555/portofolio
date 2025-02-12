"use client"

import Image from "next/image"
import { motion, useMotionValue, animate } from "framer-motion"
import { useState, useEffect } from "react"
import useMeasure from "react-use-measure"

// Photos random avec des URLs fixes pour le débogage
const photos = [
  "https://images.unsplash.com/photo-1587620962725-abab7fe55159?w=800",
  "https://images.unsplash.com/photo-1516116216624-53e697fedbea?w=800",
  "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=800",
  "https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=800",
  "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800"
]

export function PhotoCarousel() {
  const [ref, { height }] = useMeasure()
  const y = useMotionValue(0)
  
  useEffect(() => {
    const controls = animate(y, [-height, 0], {
      ease: "linear",
      duration: 30,
      repeat: Infinity,
      repeatType: "loop"
    })

    return controls.stop
  }, [height])

  return (
    <div className="w-[400px] h-screen overflow-hidden">
      <motion.div
        ref={ref}
        className="flex flex-col gap-8"
        style={{ y }}
      >
        {/* Premier set d'images */}
        <div className="flex flex-col gap-8">
          {photos.map((photo, idx) => (
            <div 
              key={idx}
              className="relative h-[300px] w-full shrink-0 rounded-xl overflow-hidden"
            >
              <Image
                src={photo}
                alt={`Photo ${idx + 1}`}
                fill
                className="object-cover"
                sizes="400px"
                priority={idx < 2}
              />
            </div>
          ))}
        </div>

        {/* Deuxième set d'images pour le défilement infini */}
        <div className="flex flex-col gap-8">
          {photos.map((photo, idx) => (
            <div 
              key={`clone-${idx}`}
              className="relative h-[300px] w-full shrink-0 rounded-xl overflow-hidden"
            >
              <Image
                src={photo}
                alt={`Photo ${idx + 1}`}
                fill
                className="object-cover"
                sizes="400px"
              />
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
} 