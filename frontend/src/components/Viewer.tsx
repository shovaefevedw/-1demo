import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { formatCount } from "../lib/format";
import type { Projection } from "../lib/types";

type Props = {
  positions: Float32Array | null;
  colors: Float32Array | null;
  pointSize: number;
  opacity: number;
  background: string;
  projection: Projection;
  fov: number;
  fitToken: number;
  updating: boolean;
  elapsedMs: number | null;
  onFit: () => void;
  onResetCamera: () => void;
};

export default function Viewer(props: Props) {
  const [fps, setFps] = useState(0);
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const count = props.positions ? props.positions.length / 3 : 0;

  return (
    <section className="relative min-h-0 min-w-0 overflow-hidden bg-[#0b0c0f]">
      <Canvas
        key={props.projection}
        orthographic={props.projection === "orthographic"}
        dpr={[1, 1.5]}
        gl={{ antialias: false, powerPreference: "high-performance", toneMapping: THREE.NoToneMapping }}
        camera={
          props.projection === "orthographic"
            ? { position: [0, 0, -4], zoom: 80, near: 0.01, far: 100 }
            : { position: [0, 0.2, -3], fov: props.fov, near: 0.01, far: 100 }
        }
        onCreated={({ camera }) => {
          camera.lookAt(0, 0, 1);
        }}
      >
        <color attach="background" args={[props.background]} />
        {props.positions && props.colors && (
          <Cloud positions={props.positions} colors={props.colors} pointSize={props.pointSize} opacity={props.opacity} />
        )}
        <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} enablePan screenSpacePanning />
        <CameraFrame
          positions={props.positions}
          fitToken={props.fitToken}
          fov={props.fov}
          controlsRef={controlsRef}
        />
        <FpsProbe onFps={setFps} />
      </Canvas>

      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 p-3">
        <div className="rounded-lg bg-black/50 px-3 py-2 text-xs text-zinc-200 backdrop-blur-sm">
          <div className="tabular-nums">{count > 0 ? `${formatCount(count)} points` : "No point cloud"}</div>
          <div className="mt-0.5 text-[11px] text-zinc-400">
            {fps > 0 ? `${fps} FPS` : "FPS —"}
            {props.elapsedMs != null ? ` · ${props.elapsedMs} ms` : ""}
            {props.updating ? " · updating" : ""}
          </div>
        </div>
        <div className="pointer-events-auto flex gap-1">
          <button type="button" className="rounded-md bg-black/55 px-2.5 py-1.5 text-xs text-zinc-100" onClick={props.onResetCamera}>
            Reset view
          </button>
          <button type="button" className="rounded-md bg-black/55 px-2.5 py-1.5 text-xs text-zinc-100" onClick={props.onFit}>
            Fit
          </button>
        </div>
      </div>

      {count === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center px-8 text-center">
          <p className="max-w-md px-8 text-sm leading-relaxed text-zinc-400">
            Upload a photograph, generate a relative depth map, then build the colored point cloud.
            Drag to orbit, scroll to zoom, and right-drag to pan.
          </p>
        </div>
      )}
    </section>
  );
}

function Cloud({
  positions,
  colors,
  pointSize,
  opacity,
}: {
  positions: Float32Array;
  colors: Float32Array;
  pointSize: number;
  opacity: number;
}) {
  const geometry = useMemo(() => {
    const next = new THREE.BufferGeometry();
    next.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    next.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    next.computeBoundingSphere();
    return next;
  }, [positions, colors]);

  useEffect(() => {
    return () => geometry.dispose();
  }, [geometry]);

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        size={pointSize}
        sizeAttenuation={false}
        vertexColors
        opacity={opacity}
        transparent={opacity < 0.999}
        depthWrite={opacity >= 0.999}
        toneMapped={false}
      />
    </points>
  );
}

function CameraFrame({
  positions,
  fitToken,
  fov,
  controlsRef,
}: {
  positions: Float32Array | null;
  fitToken: number;
  fov: number;
  controlsRef: RefObject<OrbitControlsImpl | null>;
}) {
  const camera = useThree((state) => state.camera);
  const size = useThree((state) => state.size);
  const positionsRef = useRef(positions);
  const sizeRef = useRef(size);
  positionsRef.current = positions;
  sizeRef.current = size;

  useEffect(() => {
    const current = positionsRef.current;
    if (!current || current.length < 3) return;
    let frame = 0;
    let handle = 0;
    const apply = () => {
      frameCloud(camera, controlsRef.current, current, sizeRef.current, fov);
      frame += 1;
      if (frame < 3) handle = requestAnimationFrame(apply);
    };
    apply();
    return () => cancelAnimationFrame(handle);
  }, [fitToken, camera, controlsRef, fov]);

  return null;
}

function FpsProbe({ onFps }: { onFps: (fps: number) => void }) {
  const frames = useRef(0);
  const last = useRef(performance.now());
  useFrame(() => {
    frames.current += 1;
    const now = performance.now();
    if (now - last.current >= 500) {
      onFps(Math.round((frames.current * 1000) / (now - last.current)));
      frames.current = 0;
      last.current = now;
    }
  });
  return null;
}

function frameCloud(
  camera: THREE.Camera,
  controls: OrbitControlsImpl | null,
  positions: Float32Array,
  viewport: { width: number; height: number },
  fov: number,
) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;
  const width = Math.max(maxX - minX, 1e-4);
  const height = Math.max(maxY - minY, 1e-4);
  const depth = Math.max(maxZ - minZ, 1e-4);
  const aspect = Math.max(viewport.width, 1) / Math.max(viewport.height, 1);
  const vertical = (fov * Math.PI) / 180;
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
  const distance = Math.max(
    height / 2 / Math.tan(vertical / 2),
    width / 2 / Math.tan(horizontal / 2),
    depth,
  ) * 1.35;

  if (controls) {
    controls.target.set(centerX, centerY, centerZ);
  }

  if (camera instanceof THREE.PerspectiveCamera) {
    camera.fov = fov;
    camera.position.set(centerX, centerY, minZ - distance);
    camera.near = Math.max(distance / 200, 0.01);
    camera.far = distance + depth * 4 + 20;
    camera.lookAt(centerX, centerY, centerZ);
    camera.updateProjectionMatrix();
  } else if (camera instanceof THREE.OrthographicCamera) {
    const zoom = Math.min(viewport.width / width, viewport.height / height) * 0.82;
    camera.zoom = Number.isFinite(zoom) && zoom > 0 ? zoom : 1;
    camera.position.set(centerX, centerY, minZ - Math.max(depth, 1) * 4);
    camera.near = 0.01;
    camera.far = Math.max(1000, depth * 20);
    camera.lookAt(centerX, centerY, centerZ);
    camera.updateProjectionMatrix();
  }

  controls?.update();
}
