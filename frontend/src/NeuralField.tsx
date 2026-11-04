import { useEffect, useRef } from 'react';

type NeuralFieldProps = { variant?: 'hero' | 'detail' };

const palette = ['#069ad6', '#395bdf', '#7c57d8', '#d476bb', '#e6a489'];

export default function NeuralField({ variant = 'hero' }: NeuralFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let lastDraw = 0;
    let running = true;
    let visible = true;
    let pointerX = .5;
    let pointerY = .5;
    let pointerTargetX = .5;
    let pointerTargetY = .5;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const resize = () => {
      const box = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = box.width;
      height = box.height;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(0);
    };

    const point = (line: number, progress: number, time: number) => {
      const center = (line - 27.5) / 27.5;
      const pinch = 1 - .79 * Math.exp(-Math.pow((progress - .52) / .19, 2));
      const wave = Math.sin(progress * 8.3 + line * .11 + time * .24) * height * .023;
      const cursor = (pointerY - .5) * height * .065 * Math.exp(-Math.pow((progress - .52) / .28, 2));
      return {
        x: width * (.015 + .97 * progress) + (pointerX - .5) * width * .018 * Math.sin(progress * Math.PI),
        y: height * (.5 + center * .39 * pinch) + wave + cursor,
      };
    };

    const draw = (time: number) => {
      if (!width || !height) return;
      context.clearRect(0, 0, width, height);
      pointerX += (pointerTargetX - pointerX) * .035;
      pointerY += (pointerTargetY - pointerY) * .035;

      const halo = context.createRadialGradient(width * .52, height * .5, 0, width * .52, height * .5, width * .39);
      halo.addColorStop(0, '#dce8ff88');
      halo.addColorStop(.48, '#f4efff55');
      halo.addColorStop(1, '#ffffff00');
      context.fillStyle = halo;
      context.fillRect(0, 0, width, height);

      context.strokeStyle = '#b3bdd322';
      context.lineWidth = 1;
      for (const radius of [height * .16, height * .31, height * .47]) {
        context.beginPath();
        context.ellipse(width * .52, height * .5, radius * 1.35, radius, 0, 0, Math.PI * 2);
        context.stroke();
      }
      context.setLineDash([2, 7]);
      context.strokeStyle = '#8191af55';
      context.beginPath();
      context.ellipse(width * .52, height * .5, height * .27, height * .27, -.22, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);

      for (let line = 0; line < 56; line++) {
        const color = palette[Math.min(palette.length - 1, Math.floor(line / 11.2))];
        context.beginPath();
        for (let step = 0; step <= 92; step++) {
          const p = point(line, step / 92, time);
          if (step === 0) context.moveTo(p.x, p.y);
          else context.lineTo(p.x, p.y);
        }
        context.strokeStyle = `${color}72`;
        context.lineWidth = line % 7 === 0 ? 1.25 : .7;
        context.stroke();

        for (let dot = 0; dot < 3; dot++) {
          const progress = (line * .117 + dot * .337 + time * (.016 + (line % 4) * .002)) % 1;
          const p = point(line, progress, time);
          const glow = dot === 0 ? 2.7 : 1.6;
          context.beginPath();
          context.arc(p.x, p.y, glow, 0, Math.PI * 2);
          context.fillStyle = color;
          context.fill();
        }
      }

      const center = point(28, .52, time);
      const pulse = 5 + Math.sin(time * 1.5) * 1.5;
      context.beginPath();
      context.arc(center.x, center.y, pulse + 9, 0, Math.PI * 2);
      context.fillStyle = '#7167d922';
      context.fill();
      context.beginPath();
      context.arc(center.x, center.y, pulse, 0, Math.PI * 2);
      context.fillStyle = '#7762d8';
      context.fill();
    };

    const tick = (now: number) => {
      if (!running) return;
      if (visible && now - lastDraw >= 32) {
        draw(now / 1000);
        lastDraw = now;
      }
      frame = window.requestAnimationFrame(tick);
    };
    const onPointerMove = (event: PointerEvent) => {
      const box = canvas.getBoundingClientRect();
      pointerTargetX = (event.clientX - box.left) / box.width;
      pointerTargetY = (event.clientY - box.top) / box.height;
    };
    const observer = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? true; }, { rootMargin: '120px' });
    observer.observe(canvas);
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    canvas.addEventListener('pointermove', onPointerMove);
    resize();
    if (!reducedMotion) frame = window.requestAnimationFrame(tick);
    return () => {
      running = false;
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      resizeObserver.disconnect();
      canvas.removeEventListener('pointermove', onPointerMove);
    };
  }, [variant]);

  return <canvas ref={canvasRef} className={`neural-field neural-field--${variant}`} aria-hidden="true" />;
}
