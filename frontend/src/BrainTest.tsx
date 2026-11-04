import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as nifti from 'nifti-reader-js';
import brainModelUrl from './assets/brain.glb?url';
import './brain.css';

type Channel = { name: string; value: number };
type Volume = { name: string; width: number; height: number; depth: number; datatype: number; littleEndian: boolean; bits: number; slope: number; intercept: number; image: ArrayBuffer };
type Slice = { width: number; height: number; pixels: Uint8Array };
type Viewer = { renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; controls: OrbitControls; brain: THREE.Group; brainMeshes: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>[]; sliceMesh: THREE.Mesh | null; resize: ResizeObserver };

const POSITIONS: Record<string, [number, number, number]> = {
  FP1: [-0.45, 0.67, 0.76], FP2: [0.45, 0.67, 0.76],
  F7: [-0.92, 0.42, 0.46], F3: [-0.5, 0.55, 0.62], FZ: [0, 0.82, 0.5], F4: [0.5, 0.55, 0.62], F8: [0.92, 0.42, 0.46],
  T7: [-1.07, 0.05, 0.08], C3: [-0.67, 0.27, 0.64], CZ: [0, 0.88, 0.13], C4: [0.67, 0.27, 0.64], T8: [1.07, 0.05, 0.08],
  P7: [-0.88, -0.25, -0.42], P3: [-0.45, 0.26, -0.64], PZ: [0, 0.72, -0.55], P4: [0.45, 0.26, -0.64], P8: [0.88, -0.25, -0.42],
  O1: [-0.38, -0.2, -0.85], OZ: [0, -0.15, -0.92], O2: [0.38, -0.2, -0.85],
};
const PREVIEW_CHANNELS: Channel[] = [
  { name: 'Fp1', value: .56 }, { name: 'Fp2', value: .72 }, { name: 'F3', value: .83 }, { name: 'F4', value: .64 },
  { name: 'C3', value: .48 }, { name: 'C4', value: .69 }, { name: 'P3', value: .39 }, { name: 'P4', value: .52 },
  { name: 'O1', value: .31 }, { name: 'O2', value: .44 },
];
const SUPPORTED_TYPES = new Set([2, 4, 8, 16, 64, 256, 512, 768]);

function parseEegCsv(text: string): Channel[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const header = lines.shift()?.split(',').map(value => value.trim().replace(/^"|"$/g, '').toLowerCase()) || [];
  const channelIndex = header.findIndex(value => ['channel', 'electrode'].includes(value));
  const valueIndex = header.findIndex(value => ['value', 'amplitude', 'amplitude_uv', 'amplitude_µv'].includes(value));
  if (channelIndex < 0 || valueIndex < 0) throw new Error('Use a CSV with channel,value columns.');
  const channels = new Map<string, number>();
  for (const line of lines.slice(0, 256)) {
    const cells = line.split(',').map(value => value.trim().replace(/^"|"$/g, ''));
    const name = cells[channelIndex]?.toUpperCase();
    const value = Number(cells[valueIndex]);
    if (name && name in POSITIONS && Number.isFinite(value)) channels.set(name, value);
  }
  if (!channels.size) throw new Error('No supported 10–20 channels with numeric values were found.');
  return Array.from(channels, ([name, value]) => ({ name, value }));
}

async function readVolume(file: File): Promise<Volume> {
  if (file.size > 96 * 1024 * 1024) throw new Error('Choose a NIfTI file under 96 MB for this browser preview.');
  const bytes = await file.arrayBuffer();
  const data = nifti.isCompressed(bytes) ? await nifti.decompressAsync(bytes) : bytes;
  if (!nifti.isNIFTI(data)) throw new Error('This is not a valid .nii or .nii.gz file.');
  const header = nifti.readHeader(data);
  const [width, height, depth] = header.dims.slice(1, 4);
  if (!width || !height || !depth || width > 512 || height > 512 || depth > 512) throw new Error('The scan dimensions are outside this preview’s supported range.');
  if ((header.dims[4] || 1) > 1) throw new Error('This preview supports 3D scans; 4D time series are not supported yet.');
  if (!SUPPORTED_TYPES.has(header.datatypeCode)) throw new Error(`NIfTI datatype ${header.datatypeCode} is not supported in this preview.`);
  if (width * height * depth * header.numBitsPerVoxel / 8 > 256 * 1024 * 1024) throw new Error('The uncompressed scan exceeds the 256 MB preview limit.');
  const image = nifti.readImage(header, data);
  if (image.byteLength > 256 * 1024 * 1024) throw new Error('The uncompressed scan exceeds the 256 MB preview limit.');
  return { name: file.name, width, height, depth, datatype: header.datatypeCode, littleEndian: header.littleEndian, bits: header.numBitsPerVoxel, slope: Number.isFinite(header.scl_slope) && header.scl_slope !== 0 ? header.scl_slope : 1, intercept: Number.isFinite(header.scl_inter) ? header.scl_inter : 0, image };
}

function voxel(view: DataView, index: number, volume: Volume): number {
  const offset = index * volume.bits / 8;
  let raw: number;
  switch (volume.datatype) {
    case 2: raw = view.getUint8(offset); break;
    case 4: raw = view.getInt16(offset, volume.littleEndian); break;
    case 8: raw = view.getInt32(offset, volume.littleEndian); break;
    case 16: raw = view.getFloat32(offset, volume.littleEndian); break;
    case 64: raw = view.getFloat64(offset, volume.littleEndian); break;
    case 256: raw = view.getInt8(offset); break;
    case 512: raw = view.getUint16(offset, volume.littleEndian); break;
    case 768: raw = view.getUint32(offset, volume.littleEndian); break;
    default: raw = 0;
  }
  return raw * volume.slope + volume.intercept;
}

function makeSlice(volume: Volume, z: number): Slice {
  const { width, height } = volume;
  const view = new DataView(volume.image);
  const values = new Float32Array(width * height);
  const ordered: number[] = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const value = voxel(view, (z * height + y) * width + x, volume);
    values[y * width + x] = value;
    if (Number.isFinite(value)) ordered.push(value);
  }
  ordered.sort((a, b) => a - b);
  const low = ordered[Math.floor(ordered.length * .02)] ?? 0;
  const high = ordered[Math.floor(ordered.length * .98)] ?? 1;
  const range = Math.max(high - low, 1e-6);
  const pixels = new Uint8Array(width * height * 4);
  for (let i = 0; i < values.length; i++) {
    const shade = Math.max(0, Math.min(255, Math.round((values[i] - low) / range * 255)));
    const target = ((height - 1 - Math.floor(i / width)) * width + (i % width)) * 4;
    pixels[target] = shade; pixels[target + 1] = shade; pixels[target + 2] = shade; pixels[target + 3] = shade < 3 ? 0 : 235;
  }
  return { width, height, pixels };
}

function makeLobe(side: -1 | 1, isCerebellum = false): THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial> {
  const geometry = new THREE.SphereGeometry(1, 92, 64);
  const position = geometry.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
    const u = Math.atan2(z, x), v = Math.asin(y);
    const folds = Math.sin(18 * u + 2.5 * Math.sin(5 * v)) * Math.sin(14 * v + 1.6 * Math.sin(6 * u));
    const fine = Math.sin(27 * u - 8 * v) * Math.sin(21 * v + 4 * u);
    const radius = 1 + (isCerebellum ? .065 : .048) * folds + .015 * fine;
    if (isCerebellum) position.setXYZ(i, x * radius * .54, y * radius * .31 - .68, z * radius * .42 - .34);
    else position.setXYZ(i, x * radius * .58 + side * .56, y * radius * .82 + .13, z * radius * .9);
  }
  geometry.computeVertexNormals();
  geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(position.count * 3), 3));
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .93, metalness: 0, transparent: true, side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

function prepareBrainModel(scene: THREE.Group): Viewer['brainMeshes'] {
  scene.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(scene);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const scale = 2.45 / Math.max(size.x, size.y, size.z);
  const meshes: Viewer['brainMeshes'] = [];
  scene.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return;
    const geometry = (object.geometry as THREE.BufferGeometry).clone();
    geometry.applyMatrix4(object.matrixWorld);
    geometry.translate(-center.x, -center.y, -center.z);
    geometry.scale(scale, scale, scale);
    geometry.computeVertexNormals();
    const position = geometry.getAttribute('position');
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(position.count * 3), 3));
    const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .93, metalness: 0, transparent: true, side: THREE.DoubleSide });
    meshes.push(new THREE.Mesh(geometry, material));
  });
  return meshes;
}

function paintBrain(meshes: Viewer['brainMeshes'], channels: Channel[], showEeg: boolean, gain: number, mriActive: boolean) {
  const peak = Math.max(...channels.map(channel => Math.abs(channel.value)), .0001);
  const base = new THREE.Color('#ecebea');
  const hot = new THREE.Color('#ff786d');
  for (const mesh of meshes) {
    const position = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const colors = mesh.geometry.getAttribute('color') as THREE.BufferAttribute;
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i), y = position.getY(i), z = position.getZ(i);
      let signal = 0;
      if (showEeg) for (const channel of channels) {
        const spot = POSITIONS[channel.name.toUpperCase()];
        if (!spot) continue;
        const distance = (x - spot[0]) ** 2 + (y - spot[1]) ** 2 + (z - spot[2]) ** 2;
        signal = Math.max(signal, Math.abs(channel.value) / peak * Math.exp(-distance / .16));
      }
      const texture = .78 + .22 * Math.sin(x * 31 + z * 19) * Math.sin(y * 25 - x * 17);
      const color = base.clone().lerp(hot, Math.min(.85, Math.max(0, signal * gain * texture)));
      colors.setXYZ(i, color.r, color.g, color.b);
    }
    colors.needsUpdate = true;
    mesh.material.opacity = mriActive ? .48 : 1;
    mesh.material.depthWrite = !mriActive;
  }
}

function BrainViewport({ channels, showEeg, gain, slice, resetKey }: { channels: Channel[]; showEeg: boolean; gain: number; slice: Slice | null; resetKey: number }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [error, setError] = useState('');
  const [modelVersion, setModelVersion] = useState(0);
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); }
    catch { setError('3D rendering is unavailable in this browser.'); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor('#111315');
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 30);
    camera.position.set(.05, .12, 4.3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -.02, 0);
    controls.enableDamping = true;
    controls.dampingFactor = .07;
    controls.enablePan = false;
    controls.minDistance = 2.8;
    controls.maxDistance = 7;
    const ambient = new THREE.AmbientLight('#ffffff', 1.8);
    const key = new THREE.DirectionalLight('#ffffff', 3.2); key.position.set(-2, 3, 5);
    const rim = new THREE.DirectionalLight('#d5d7e6', 1.5); rim.position.set(2, 1, -4);
    scene.add(ambient, key, rim);
    const brain = new THREE.Group();
    const brainMeshes = [makeLobe(-1), makeLobe(1), makeLobe(-1, true)];
    for (const mesh of brainMeshes) brain.add(mesh);
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(.105, .15, .39, 20), new THREE.MeshStandardMaterial({ color: '#d9d8d6', roughness: .94 }));
    stem.position.set(0, -.93, -.22); brain.add(stem);
    brain.rotation.y = -.13;
    scene.add(brain);
    const resize = new ResizeObserver(() => {
      const width = Math.max(host.clientWidth, 1), height = Math.max(host.clientHeight, 1);
      renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
    });
    resize.observe(host);
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
    viewerRef.current = { renderer, camera, controls, brain, brainMeshes, sliceMesh: null, resize };
    let cancelled = false;
    new GLTFLoader().load(brainModelUrl, gltf => {
      if (cancelled || !viewerRef.current) return;
      const loadedMeshes = prepareBrainModel(gltf.scene);
      if (!loadedMeshes.length) { setError('The brain model did not contain a visible mesh.'); return; }
      for (const child of [...brain.children]) {
        if (child === viewerRef.current.sliceMesh) continue;
        brain.remove(child);
        if (child instanceof THREE.Mesh) { child.geometry.dispose(); (child.material as THREE.Material).dispose(); }
      }
      for (const mesh of loadedMeshes) brain.add(mesh);
      viewerRef.current.brainMeshes = loadedMeshes;
      setModelVersion(value => value + 1);
    }, undefined, () => setError('The anatomical preview model could not be loaded.'));
    return () => {
      cancelled = true;
      resize.disconnect(); renderer.setAnimationLoop(null); controls.dispose();
      brain.traverse(object => { if (object instanceof THREE.Mesh) { object.geometry.dispose(); const materials = Array.isArray(object.material) ? object.material : [object.material]; for (const material of materials) { if (material instanceof THREE.MeshBasicMaterial) material.map?.dispose(); material.dispose(); } } });
      renderer.dispose(); renderer.domElement.remove(); viewerRef.current = null;
    };
  }, []);
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    paintBrain(viewer.brainMeshes, channels, showEeg, gain, !!slice);
    if (viewer.sliceMesh) { viewer.brain.remove(viewer.sliceMesh); viewer.sliceMesh.geometry.dispose(); const material = viewer.sliceMesh.material as THREE.MeshBasicMaterial; material.map?.dispose(); material.dispose(); viewer.sliceMesh = null; }
    if (slice) {
      const texture = new THREE.DataTexture(slice.pixels, slice.width, slice.height, THREE.RGBAFormat);
      texture.colorSpace = THREE.SRGBColorSpace; texture.needsUpdate = true;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.85, 1.65), new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: .9, side: THREE.DoubleSide, depthWrite: false }));
      mesh.rotation.x = -Math.PI / 2; mesh.position.y = .02; mesh.renderOrder = 2;
      viewer.brain.add(mesh); viewer.sliceMesh = mesh;
    }
  }, [channels, showEeg, gain, slice, modelVersion]);
  const hasSlice = !!slice;
  useEffect(() => { const viewer = viewerRef.current; if (!viewer) return; viewer.camera.position.set(hasSlice ? 2.05 : .05, hasSlice ? 1.6 : .12, hasSlice ? 3.75 : 4.3); viewer.controls.target.set(0, -.02, 0); viewer.controls.update(); }, [hasSlice, resetKey]);
  return <div className="brain-viewport" ref={hostRef} role="img" aria-label="Rotatable illustrative 3D brain with EEG signal overlay and optional unregistered MRI voxel slice">{error && <p className="brain-render-error">{error}</p>}</div>;
}

export default function BrainTest() {
  const [channels, setChannels] = useState<Channel[]>(PREVIEW_CHANNELS);
  const [eegName, setEegName] = useState('Illustrative sample');
  const [eegError, setEegError] = useState('');
  const [volume, setVolume] = useState<Volume | null>(null);
  const [mriError, setMriError] = useState('');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [showEeg, setShowEeg] = useState(true);
  const [gain, setGain] = useState(1);
  const [resetKey, setResetKey] = useState(0);
  const slice = useMemo(() => volume ? makeSlice(volume, sliceIndex) : null, [volume, sliceIndex]);
  const sliceCanvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!slice || !sliceCanvas.current) return;
    const canvas = sliceCanvas.current; canvas.width = slice.width; canvas.height = slice.height;
    const context = canvas.getContext('2d'); if (!context) return;
    context.putImageData(new ImageData(new Uint8ClampedArray(slice.pixels), slice.width, slice.height), 0, 0);
  }, [slice]);
  async function loadEeg(file?: File) {
    if (!file) return;
    setEegError('');
    try { const data = parseEegCsv(await file.text()); setChannels(data); setEegName(file.name); setShowEeg(true); }
    catch (error) { setEegError(String(error instanceof Error ? error.message : error)); }
  }
  async function loadMri(file?: File) {
    if (!file) return;
    setMriError('');
    try { const data = await readVolume(file); setVolume(data); setSliceIndex(Math.floor(data.depth / 2)); }
    catch (error) { setMriError(String(error instanceof Error ? error.message : error)); }
  }
  const peak = Math.max(...channels.map(channel => Math.abs(channel.value)), .0001);
  return <div className="brain-test">
    <div className="brain-stage">
      <div className="brain-stage-header"><div><span className="brain-stage-kicker">VISUALIZATION TEST</span><strong>Brain view</strong></div><div className="brain-stage-actions"><span className="brain-stage-status">{volume ? 'MRI voxel slice loaded' : 'Illustrative anatomy'}</span><button type="button" onClick={() => setResetKey(value => value + 1)}>Reset view</button></div></div>
      <BrainViewport channels={channels} showEeg={showEeg} gain={gain} slice={slice} resetKey={resetKey}/>
      <div className="brain-stage-footer"><span>Drag to rotate · Scroll to zoom</span><span>{eegName === 'Illustrative sample' ? 'Preview signal · not measured' : `${channels.length} EEG channels · local file`}</span></div>
    </div>
    <div className="brain-controls"><div className="brain-control-heading"><div><h2>Layers</h2><p>Explore the structural surface and relative EEG amplitude.</p></div><span>Local browser preview</span></div><div className="brain-control-row"><label className="brain-toggle"><input type="checkbox" checked={showEeg} onChange={event => setShowEeg(event.target.checked)}/><span>EEG overlay</span></label><label className="brain-gain">Overlay strength <input aria-label="Overlay strength" type="range" min="0" max="2" step="0.1" value={gain} onChange={event => setGain(Number(event.target.value))}/></label></div></div>
    <div className="brain-data-grid"><section className="brain-data-card"><div className="brain-data-head"><div><h2>EEG channels</h2><p>{eegName}</p></div><label className="button secondary brain-file-button">Import CSV<input type="file" accept=".csv,text/csv" onChange={event => loadEeg(event.target.files?.[0])}/></label></div><p className="brain-help">CSV columns: <code>channel,value</code>. Supports standard 10–20 labels. Bars show magnitude relative to this file’s peak, not attention or learning.</p>{eegError && <p className="brain-error" role="alert">{eegError}</p>}<div className="brain-channel-list">{channels.slice(0, 12).map(channel => <div className="brain-channel" key={channel.name}><span>{channel.name}</span><div className="brain-channel-track"><span style={{ width: `${Math.round(Math.abs(channel.value) / peak * 100)}%` }}/></div><strong>{Math.round(Math.abs(channel.value) / peak * 100)}%</strong></div>)}</div>{eegName !== 'Illustrative sample' && <button className="brain-text-button" onClick={() => { setChannels(PREVIEW_CHANNELS); setEegName('Illustrative sample'); setEegError(''); }}>Return to preview signal</button>}</section>
      <section className="brain-data-card"><div className="brain-data-head"><div><h2>MRI slice</h2><p>{volume ? volume.name : 'No scan loaded'}</p></div><label className="button secondary brain-file-button">Import NIfTI<input type="file" accept=".nii,.nii.gz" onChange={event => loadMri(event.target.files?.[0])}/></label></div><p className="brain-help">Load a .nii or .nii.gz scan to inspect a voxel-axis slice. The plane in the 3D view is not registered to the illustrative surface.</p>{mriError && <p className="brain-error" role="alert">{mriError}</p>}{volume && slice ? <><div className="brain-mri-preview"><canvas ref={sliceCanvas} aria-label={`MRI voxel slice ${sliceIndex + 1} of ${volume.depth}`}/></div><label className="brain-slice-control">Slice {sliceIndex + 1} / {volume.depth}<input type="range" min="0" max={volume.depth - 1} value={sliceIndex} onChange={event => setSliceIndex(Number(event.target.value))}/></label><div className="brain-mri-meta">{volume.width} × {volume.height} × {volume.depth} voxels <button className="brain-text-button" onClick={() => { setVolume(null); setMriError(''); }}>Clear scan</button></div></> : <div className="brain-mri-empty">Import an MRI volume to preview its slices here.</div>}</section></div>
    <p className="brain-caveat">This is a visualization prototype. The default signal and surface are illustrative. Imported files stay in this browser session; no clinical interpretation, source localization, or measured learning claim is produced.</p>
  </div>;
}
