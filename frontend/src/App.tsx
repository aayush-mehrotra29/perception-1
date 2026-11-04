import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Activity, ArrowRight, ArrowUpRight, Brain, FileText, Globe2, LayoutGrid, PanelLeft, Plus, Search } from 'lucide-react';
import { api, json, time } from './api';
import perceptionMark from './assets/perception-mark.svg';
import Landing from './Landing';
import { LearnerSetup, LessonOverview, QuestionEditor } from './Preparation';
import type { Finding, Item, Job, Lesson, Profile, Report, Run } from './types';

type View = 'Landing' | 'Projects' | 'NewLesson' | 'Overview' | 'Lesson' | 'Questions' | 'Learners' | 'Simulation' | 'Findings' | 'Evidence' | 'Test';
const views: View[] = ['Projects', 'Overview', 'Lesson', 'Questions', 'Learners', 'Simulation', 'Findings', 'Evidence', 'Test'];
const workspaceTabs: { view: View; label: string }[] = [
  { view: 'Overview', label: 'Overview' }, { view: 'Lesson', label: 'Source' },
  { view: 'Questions', label: 'Questions' }, { view: 'Learners', label: 'Learners' },
  { view: 'Simulation', label: 'Runs' }, { view: 'Findings', label: 'Findings' },
  { view: 'Evidence', label: 'Evidence' }, { view: 'Test', label: 'Lab' },
];
const BrainTest = lazy(() => import('./BrainTest'));

function NavIcon({ view }: { view: View }) {
  const props = { size: 16, strokeWidth: 1.7, 'aria-hidden': true as const };
  if (view === 'Projects') return <LayoutGrid {...props}/>;
  if (view === 'Lesson') return <FileText {...props}/>;
  if (view === 'Simulation') return <Globe2 {...props}/>;
  if (view === 'Findings') return <Search {...props}/>;
  if (view === 'Test') return <Brain {...props}/>;
  return <Activity {...props}/>;
}

function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'blue' | 'amber' | 'green' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function WorkspaceTabs({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return <nav className="section-tabs" aria-label="Lesson sections">{workspaceTabs.map(tab => <button key={tab.view} className={`section-tab ${view === tab.view ? 'active' : ''} ${tab.view === 'Test' ? 'lab-tab' : ''}`} aria-current={view === tab.view ? 'page' : undefined} onClick={() => onChange(tab.view)}>{tab.label}</button>)}</nav>;
}

function LessonComposer({ onDone, onCancel }: { onDone: (id: string, job: string) => Promise<void>; onCancel: () => void }) {
  const [title, setTitle] = useState('');
  const [transcript, setTranscript] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [items, setItems] = useState<Item[]>([{ id: 'q1', prompt: '', options: ['', ''], correct_index: 0 }]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  function update(index: number, patch: Partial<Item>) { setItems(current => current.map((item, i) => i === index ? { ...item, ...patch } : item)); }
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const form = new FormData();
      form.append('title', title); form.append('transcript', transcript); form.append('assessment', JSON.stringify(items));
      if (!file) throw new Error('Choose a video.');
      form.append('video', file);
      const result = await api<{lesson: Lesson; job_id: string}>('/lessons', { method: 'POST', body: form });
      await onDone(result.lesson.id, result.job_id);
    } catch (e) { setError(String(e instanceof Error ? e.message : e)); }
    finally { setBusy(false); }
  }
  return <section className="lesson-composer" aria-labelledby="new-lesson-heading">
    <div className="page-head"><div><h1 id="new-lesson-heading">Add lesson</h1><p>Upload a short lesson and add questions to create a new simulation workspace.</p></div><button type="button" className="button secondary" onClick={onCancel}>Back to lessons</button></div>
    <form onSubmit={submit} className="form-stack lesson-composer-form">
      <section className="composer-panel"><div className="composer-section-heading"><span className="eyebrow">01 / SOURCE</span><h2>Lesson material</h2></div><div className="composer-source-fields">
        <label>Lesson title<input required maxLength={160} value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g. The ideal rocket equation" /></label>
        <label>Video, 10 minutes or less<input required type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
        <label>Transcript <span className="optional">optional at upload</span><textarea rows={5} value={transcript} onChange={e => setTranscript(e.target.value)} placeholder={'00:00 - 00:24 | Today we will…\n00:24 - 00:48 | Next…'} /><small>Use one timed passage per line for video linking. Untimed lines remain passage linked.</small></label>
      </div></section>
      <section className="composer-panel"><div className="composer-section-heading row between"><div><span className="eyebrow">02 / ASSESSMENT</span><h2>Check for understanding</h2></div><button type="button" className="button secondary" onClick={() => setItems([...items, { id: `q${items.length+1}`, prompt: '', options: ['', ''], correct_index: 0 }])}><Plus size={13} strokeWidth={1.7} aria-hidden="true"/>Add question</button></div>
        {items.map((item, index) => <div className="item-editor" key={index}><label>Question {index + 1}<input required value={item.prompt} onChange={e => update(index, { prompt: e.target.value })} placeholder="What should the learner understand?" /></label><div className="option-grid">{item.options.map((option, o) => <label key={o}>Option {o + 1}<input required value={option} onChange={e => update(index, { options: item.options.map((v, i) => i === o ? e.target.value : v) })} /></label>)}</div><label>Correct option<select value={item.correct_index ?? 0} onChange={e => update(index, { correct_index: Number(e.target.value) })}>{item.options.map((_, i) => <option key={i} value={i}>Option {i + 1}</option>)}</select></label></div>)}
      </section>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="row end composer-actions"><button type="button" className="button secondary" onClick={onCancel}>Cancel</button><button disabled={busy} className="button primary">{busy ? 'Uploading…' : 'Analyze lesson'}</button></div>
    </form>
  </section>;
}

export default function App() {
  const [view, setView] = useState<View>(() => window.location.hash === '#workspace' ? 'Projects' : 'Landing');
  const [projects, setProjects] = useState<Lesson[]>([]);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [report, setReport] = useState<Report>({});
  const [job, setJob] = useState<Job | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [assessment, setAssessment] = useState<Item[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profilesSaved, setProfilesSaved] = useState(false);
  const [activeProfileCount, setActiveProfileCount] = useState(12);
  const [seeds, setSeeds] = useState(3);
  const [maxActions, setMaxActions] = useState(20);
  const [maxTokens, setMaxTokens] = useState(500);
  const [maxCost, setMaxCost] = useState(5);
  const [transcriptDraft, setTranscriptDraft] = useState('');
  const [conceptDraft, setConceptDraft] = useState('');
  const [parentLesson, setParentLesson] = useState<Lesson | null>(null);
  const [parentRuns, setParentRuns] = useState<Run[]>([]);
  const [pendingSeek, setPendingSeek] = useState<number | null>(null);
  const [research, setResearch] = useState<{research_volume_mounted:boolean;internal_free_bytes:number;internal_reserve_bytes:number;numerical_execution_allowed:boolean} | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const sidebarLinksRef = useRef<HTMLDivElement>(null);
  const [sidebarSelection, setSidebarSelection] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [sidebarSelectionReady, setSidebarSelectionReady] = useState(false);
  const activeSidebarKey = view === 'Landing' || view === 'Projects' || view === 'NewLesson' ? 'nav-projects' : view === 'Simulation' ? 'nav-simulation' : lesson ? `recent-${lesson.id}` : 'nav-simulation';

  useEffect(() => {
    const syncHash = () => setView(current => window.location.hash === '#workspace' ? (current === 'Landing' ? 'Projects' : current) : 'Landing');
    window.addEventListener('hashchange', syncHash);
    return () => window.removeEventListener('hashchange', syncHash);
  }, []);
  function openWorkspace() { window.location.hash = 'workspace'; setView('Projects'); }
  function openLanding() { window.history.pushState(null, '', window.location.pathname + window.location.search); setView('Landing'); }

  useLayoutEffect(() => {
    const container = sidebarLinksRef.current;
    if (!container) return;
    const selected = container.querySelector<HTMLElement>('.nav-item.active, .recent-item.active');
    const measure = () => {
      if (!selected) { setSidebarSelection(null); return; }
      const parent = container.getBoundingClientRect();
      const row = selected.getBoundingClientRect();
      setSidebarSelection({ x: row.left - parent.left, y: row.top - parent.top, width: row.width, height: row.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    if (selected) observer.observe(selected);
    window.addEventListener('resize', measure);
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); };
  }, [activeSidebarKey, sidebarCollapsed, projects.length, view]);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setSidebarSelectionReady(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  const reload = useCallback(async (id?: string) => {
    const freshProjects = await api<Lesson[]>('/projects'); setProjects(freshProjects);
    if (id) {
      const [l, r, f, e, a, p] = await Promise.all([
        api<Lesson>(`/lessons/${id}`), api<Run[]>(`/simulations/${id}`),
        api<Finding[]>(`/findings/${id}`), api<Report>(`/reports/${id}`),
        api<Item[]>(`/lessons/${id}/assessment`), api<Profile[]>(`/lessons/${id}/profiles`)]);
      setLesson(l); setRuns(r); setFindings(f); setReport(e); setAssessment(a); setProfiles(p); setProfilesSaved(false);
      setConceptDraft(l.concepts.join(', '));
      setTranscriptDraft(l.transcript.map(s => s.start == null ? s.text : `${time(s.start)} - ${time(s.end)} | ${s.text}`).join('\n'));
    }
  }, []);
  useEffect(() => { reload().catch(e => setError(String(e))); api<{research:NonNullable<typeof research>}>('/health').then(x => setResearch(x.research)).catch(() => {}); }, [reload]);
  useEffect(() => {
    if (!lesson?.parent_id) { setParentLesson(null); setParentRuns([]); return; }
    Promise.all([api<Lesson>(`/lessons/${lesson.parent_id}`), api<Run[]>(`/simulations/${lesson.parent_id}`)])
      .then(([l, r]) => { setParentLesson(l); setParentRuns(r); }).catch(e => setError(String(e)));
  }, [lesson?.parent_id]);
  useEffect(() => {
    if (view !== 'Lesson' || pendingSeek == null || !videoRef.current) return;
    const video = videoRef.current;
    const target = pendingSeek;
    const jump = () => { video.currentTime = target; video.play().catch(() => {}); setPendingSeek(null); };
    if (video.readyState >= 1) jump();
    else video.addEventListener('loadedmetadata', jump, { once: true });
    return () => video.removeEventListener('loadedmetadata', jump);
  }, [view, pendingSeek, lesson?.id]);
  useEffect(() => { window.scrollTo({ top: 0, left: 0 }); }, [view, lesson?.id]);
  useEffect(() => {
    if (!job) return;
    const source = new EventSource(`/api/jobs/${job.id}/events`);
    let finished = false;
    const refresh = () => api<Job>(`/jobs/${job.id}`).then(j => {
      if (finished) return;
      setJob(j);
      if (['complete', 'failed', 'cancelled'].includes(j.state)) { finished = true; source.close(); if (lesson) reload(lesson.id).catch(e => setError(String(e))); }
    }).catch(e => setError(String(e)));
    source.addEventListener('progress', refresh);
    source.addEventListener('queued', refresh);
    source.onerror = () => { source.close(); refresh(); };
    return () => source.close();
  }, [job?.id, lesson?.id, reload]);
  async function select(id: string, target: View = 'Overview') { setError(''); await reload(id); setView(target); }
  function seek(value: number | null) { if (value != null) setPendingSeek(value); setView('Lesson'); }
  async function simulate() { if (!lesson) return; setBusy(true); setError(''); try { const result = await api<{job_id: string}>('/simulations', json('POST', { lesson_id: lesson.id, profiles, profile_count: activeProfileCount, seeds_per_profile: seeds, max_actions_per_run: maxActions, max_tokens_per_run: maxTokens, max_cost_usd_per_run: maxCost })); setJob(await api<Job>(`/jobs/${result.job_id}`)); setView('Simulation'); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  async function saveProfiles() { if (!lesson) return; setBusy(true); setError(''); try { setProfiles(await api<Profile[]>(`/lessons/${lesson.id}/profiles`, json('PUT', profiles))); setProfilesSaved(true); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  async function makeRevision(findingId: string) { setBusy(true); setError(''); try { const revised = await api<Lesson>('/revisions', json('POST', { finding_id: findingId })); await select(revised.id, 'Overview'); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  async function saveAssessment() { if (!lesson) return; setBusy(true); setError(''); try { const revised = await api<Lesson>(`/lessons/${lesson.id}/assessment`, json('PUT', assessment)); await select(revised.id, 'Questions'); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  async function saveTranscript() { if (!lesson) return; setBusy(true); setError(''); try { const result = await api<{job_id: string;lesson:Lesson}>(`/lessons/${lesson.id}/transcript`, json('POST', { transcript: transcriptDraft })); await select(result.lesson.id, 'Lesson'); setJob(await api<Job>(`/jobs/${result.job_id}`)); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  async function saveConcepts() { if (!lesson) return; setBusy(true); setError(''); try { const concepts = conceptDraft.split(',').map(x => x.trim()).filter(Boolean); const revised = await api<Lesson>(`/lessons/${lesson.id}/concepts`, json('PUT', { concepts })); await select(revised.id, 'Lesson'); } catch(e) { setError(String(e)); } finally { setBusy(false); } }
  const activeRun = runs.find(r => r.id === selectedRun) || runs[0];
  const activity = ['read_section', 'review_section', 'ask_question', 'attempt_item'].map(kind => ({ name: kind.replace('_', ' '), count: runs.reduce((n, run) => n + run.actions.filter(a => a.kind === kind).length, 0) }));
  const runSetupValid = profiles.length > 0 && profiles.length <= 12 && profiles.every(profile => profile.name.trim()) && Number.isInteger(activeProfileCount) && activeProfileCount >= 1 && activeProfileCount <= profiles.length && Number.isInteger(seeds) && seeds >= 1 && seeds <= 100 && activeProfileCount * seeds <= 100 && Number.isInteger(maxActions) && maxActions >= 1 && maxActions <= 100 && Number.isInteger(maxTokens) && maxTokens >= 0 && maxTokens <= 10000 && Number.isFinite(maxCost) && maxCost >= 0 && maxCost <= 5;

  if (view === 'Landing') return <Landing onEnter={openWorkspace}/>;

  return <div className={`shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className="sidebar"><div className="brand"><button className="brand-home" type="button" aria-label="Perception home" title="Back to landing page" onClick={openLanding}><img className="brand-symbol" src={perceptionMark} alt=""/><strong className="brand-wordmark">Perception</strong></button><button className="sidebar-toggle" type="button" aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-expanded={!sidebarCollapsed} title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={() => setSidebarCollapsed(value => !value)}><PanelLeft size={14} strokeWidth={1.7} aria-hidden="true"/></button></div>
      <div className="sidebar-links" ref={sidebarLinksRef}>
        <span className={`sidebar-selection-indicator ${sidebarSelection ? 'is-visible' : ''} ${sidebarSelectionReady ? '' : 'no-motion'}`} aria-hidden="true" style={sidebarSelection ? { transform: `translate3d(${sidebarSelection.x}px, ${sidebarSelection.y}px, 0)`, width: sidebarSelection.width, height: sidebarSelection.height } : undefined}/>
        <nav aria-label="Primary navigation">{views.filter(v => v === 'Projects' || v === 'Simulation').map(v => { const key = v === 'Projects' ? 'nav-projects' : 'nav-simulation'; const label = v === 'Projects' ? 'Lessons' : 'Simulations'; return <button key={v} className={`nav-item ${activeSidebarKey === key ? 'active' : ''}`} aria-current={activeSidebarKey === key ? 'page' : undefined} title={sidebarCollapsed ? label : undefined} onClick={() => setView(v)}><span className="nav-glyph"><NavIcon view={v}/></span><span className="nav-text">{label}</span></button>; })}</nav>
        <div className="nav-label recents-label">Recents</div><div className="recent-list">{projects.slice(0, 5).map(project => { const key = `recent-${project.id}`; return <button key={project.id} className={`recent-item ${activeSidebarKey === key ? 'active' : ''}`} aria-current={activeSidebarKey === key ? 'page' : undefined} title={sidebarCollapsed ? project.title : undefined} onClick={() => select(project.id)}><span className="recent-glyph"><NavIcon view="Lesson"/></span><span className="recent-text">{project.title}</span></button>; })}{!projects.length && <span className="recent-empty">Lessons you add appear here.</span>}</div>
      </div>
      <div className="sidebar-bottom"><div className="sidebar-avatar">P</div><div className="sidebar-workspace"><strong>Local workspace</strong><span>Research prototype</span></div></div>
    </aside>
    <main className="main">
      {error && <div className="alert" role="alert">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
      {job && ['queued', 'running', 'failed'].includes(job.state) && <div className={`jobbar ${job.state === 'failed' ? 'failed' : ''}`}><span className="pulse"/><div><strong>{job.state === 'failed' ? 'Job failed' : job.stage === 'simulating' ? 'Running trajectories' : 'Processing lesson'}</strong><span>{job.error || `${job.stage} · ${job.progress}${job.stage === 'simulating' ? ' trajectories complete' : '%'}`}</span></div>{job.state === 'running' && <button className="link-button" onClick={async () => setJob(await api<Job>(`/jobs/${job.id}/cancel`, { method: 'POST' }))}>Cancel</button>}</div>}
      <div className={`workspace-body ${view === 'NewLesson' ? 'composer-workspace' : ''}`}><div className={`page ${view === 'Projects' ? 'projects-page' : view === 'Lesson' ? 'source-page' : view === 'NewLesson' ? 'new-lesson-page' : ''}`}>
      {view === 'NewLesson' && <LessonComposer onCancel={() => setView('Projects')} onDone={async (id, jobId) => { await select(id); setJob(await api<Job>(`/jobs/${jobId}`)); }}/ >}
      {view === 'Projects' && <><div className="page-head"><div><h1>Lessons</h1><p>Create a lesson simulation, inspect the learner traces, and review findings against the source.</p></div><button className="button primary" onClick={() => setView('NewLesson')}><Plus size={13} strokeWidth={1.7} aria-hidden="true"/>Add lesson</button></div>
        <div className="section-tabs"><span className="section-tab active">All lessons</span></div>
        <div className="overview"><div><span>LESSONS</span><strong>{projects.length}</strong></div><div><span>TRAJECTORIES</span><strong>{projects.reduce((total, project) => total + (project.trajectory_count || 0), 0)}</strong></div><div><span>EVIDENCE STATUS</span><strong>Unvalidated</strong></div></div>
        <div className="section-heading"><div><h2>Recent lessons</h2></div><span>{projects.length} {projects.length === 1 ? 'lesson' : 'lessons'}</span></div>
        {projects.length ? <div className="project-list">{projects.map(p => <button key={p.id} className="project-card" onClick={() => select(p.id)}><div className="project-icon"><FileText size={17} strokeWidth={1.7} aria-hidden="true"/></div><div className="project-copy"><strong>{p.title}</strong><span>Version {p.version} · {p.sections.length} sections · {p.assessment.length} assessment items</span></div><Badge tone={p.status === 'ready' ? 'green' : 'amber'}>{p.status.replace('_', ' ')}</Badge><span className="arrow"><ArrowRight size={15} strokeWidth={1.7} aria-hidden="true"/></span></button>)}</div> : <div className="empty"><div className="empty-icon"><FileText size={22} strokeWidth={1.6} aria-hidden="true"/></div><h2>Start with a short lesson</h2><p>Upload a video and assessment. A timed transcript connects findings to the source.</p><button className="button primary" onClick={() => setView('NewLesson')}>Add first lesson</button></div>}</>}

      {view === 'Overview' && (lesson ? <><div className="page-head"><div><h1>{lesson.title}</h1><p>Prepare the lesson, run simulated learners and review the evidence in one workspace.</p></div><div className="head-badges"><Badge tone={lesson.status === 'ready' ? 'green' : 'amber'}>{lesson.status.replace('_', ' ')}</Badge></div></div><WorkspaceTabs view={view} onChange={setView}/><LessonOverview lesson={lesson} questionCount={assessment.length} profileCount={profiles.length} runCount={runs.length} findingCount={findings.length} onNavigate={setView}/></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Lesson' && (lesson ? <><div className="page-head"><div><h1>Source &amp; concepts</h1><p>Inspect the video, transcript and instructional concepts for {lesson.title}.</p></div><div className="head-badges"><Badge tone={lesson.status === 'ready' ? 'green' : 'amber'}>{lesson.status.replace('_', ' ')}</Badge></div></div><WorkspaceTabs view={view} onChange={setView}/>
        {lesson.parent_id && <div className="notice"><Badge tone="blue">Revision package</Badge><span>Added explanation or checkpoint. Original video is retained; any improvement remains a hypothesis until tested with people.</span><button className="link-button" onClick={() => select(lesson.parent_id!, 'Lesson')}>Open original <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button></div>}
        <div className="lesson-grid"><div className="lesson-main"><div className="panel video-panel"><div className="panel-header"><h2>Source video</h2><span>{lesson.duration_seconds ? `${Math.round(lesson.duration_seconds)} sec` : '—'}</span></div><video ref={videoRef} controls preload="metadata" src={`/api/lessons/${lesson.id}/media`}/></div>
          <div className="panel"><div className="panel-header"><h2>Transcript & sections</h2><Badge>Source linked</Badge></div>{lesson.sections.length ? <div className="transcript-list">{lesson.sections.map(s => <button key={s.id} className="transcript-row" onClick={() => seek(s.start)}><span className="timestamp">{time(s.start)}</span><span><strong>{s.title}</strong><small>{s.text}</small>{s.added_by_revision && <Badge tone="blue">Proposed repair</Badge>}</span></button>)}</div> : <div className="panel-empty">No transcript analyzed yet.</div>}</div></div>
          <div className="lesson-side"><div className="panel"><div className="panel-header"><h2>Concepts</h2><Badge tone={lesson.concepts_reviewed ? 'green' : 'amber'}>{lesson.concepts_reviewed ? 'Educator reviewed' : 'Review needed'}</Badge></div><p className="muted smalltext">Extracted from transcript frequency. Edit the comma-separated list to confirm it.</p><div className="chips">{lesson.concepts.length ? lesson.concepts.map(c => <span className="chip" key={c}>{c}</span>) : <span className="muted">Awaiting transcript</span>}</div><textarea rows={2} aria-label="Edit concepts" value={conceptDraft} onChange={e => setConceptDraft(e.target.value)}/><button className="button secondary full" disabled={busy || lesson.status !== 'ready'} onClick={saveConcepts}>Save reviewed concepts</button></div>
            {lesson.status === 'needs_transcript' && <div className="panel"><h2>Add transcript</h2><p className="muted smalltext">Video transcription requires a configured provider. Supply a transcript to continue now.</p><textarea rows={7} value={transcriptDraft} onChange={e => setTranscriptDraft(e.target.value)}/><button disabled={busy} className="button primary" onClick={saveTranscript}>Analyze transcript</button></div>}
          </div></div></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Questions' && (lesson ? <><div className="page-head"><div><h1>Questions &amp; answer key</h1><p>Edit the educator assessment used to inspect what simulated learners attempt.</p></div><Badge>{assessment.length} {assessment.length === 1 ? 'question' : 'questions'}</Badge></div><WorkspaceTabs view={view} onChange={setView}/><QuestionEditor items={assessment} onChange={setAssessment} onSave={saveAssessment} busy={busy}/></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Learners' && (lesson ? <><div className="page-head"><div><h1>Learner setup</h1><p>Choose the prior knowledge, misconceptions and limits for the next simulation.</p></div><Badge>{profiles.length} {profiles.length === 1 ? 'profile' : 'profiles'}</Badge></div><WorkspaceTabs view={view} onChange={setView}/><LearnerSetup profiles={profiles} onProfilesChange={nextProfiles => { setProfiles(nextProfiles); setProfilesSaved(false); }} settings={{ profileCount: activeProfileCount, seeds, maxActions, maxTokens, maxCost }} onSettingsChange={patch => { if (patch.profileCount !== undefined) setActiveProfileCount(patch.profileCount); if (patch.seeds !== undefined) setSeeds(patch.seeds); if (patch.maxActions !== undefined) setMaxActions(patch.maxActions); if (patch.maxTokens !== undefined) setMaxTokens(patch.maxTokens); if (patch.maxCost !== undefined) setMaxCost(patch.maxCost); }} onSave={saveProfiles} onRun={simulate} busy={busy} saved={profilesSaved} lessonReady={lesson.status === 'ready'}/></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Simulation' && (lesson ? <><div className="page-head"><div><h1>Simulation runs</h1><p>Inspect scripted actions across matched profiles and seeds. These are not human participants.</p></div><button className="button primary" disabled={busy || lesson.status !== 'ready' || !runSetupValid} onClick={simulate}>Run {activeProfileCount} profiles × {seeds} seeds</button></div><WorkspaceTabs view={view} onChange={setView}/>
        <div className="notice"><Badge tone="amber">Authored assumption</Badge><span>Actions use restricted tools and exposed sections. No calibrated outcome probabilities or measured mastery changes are available.</span><button className="link-button" onClick={() => setView('Learners')}>Edit learner setup <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button></div>
        <div className="stat-grid"><div className="stat"><span>TRAJECTORIES</span><strong>{runs.length}</strong><small>Generated, not independent people</small></div><div className="stat"><span>LEARNER PROFILES</span><strong>{new Set(runs.map(r => r.profile.id)).size}</strong><small>Authored prior knowledge</small></div><div className="stat"><span>MODEL PREDICTIONS</span><strong>Unavailable</strong><small>Awaiting validated behavioral head</small></div></div>
        {runs.length > 0 && <div className="simulation-grid"><div className="panel"><div className="panel-header"><h2>Action trace</h2><span>{runs.length} runs</span></div><div className="run-list">{runs.map(r => <button key={r.id} className={`run-row ${activeRun?.id === r.id ? 'selected' : ''}`} onClick={() => setSelectedRun(r.id)}><span className="run-avatar">{r.profile.id.slice(-2)}</span><span><strong>{r.profile.name}</strong><small>Seed {r.seed} · {r.actions.length} actions</small></span><span className="arrow"><ArrowRight size={15} strokeWidth={1.7} aria-hidden="true"/></span></button>)}</div></div><div><div className="panel"><div className="panel-header"><h2>{activeRun?.profile.name || 'Select a run'}</h2><Badge tone="amber">Scripted</Badge></div><div className="run-meta">Prior knowledge: {activeRun?.profile.prior_knowledge} · Misconceptions: {activeRun?.profile.misconceptions.join(', ') || 'none'} · {activeRun?.tokens_used} tokens · ${activeRun?.cost_usd.toFixed(2)}</div><div className="trace">{activeRun?.actions.map((a,i) => <div className="trace-row" key={i}><div className="trace-number">{String(i+1).padStart(2,'0')}</div><div><strong>{a.kind.replace('_',' ')}</strong><p>{a.text}{a.selected_option != null && ` Option ${a.selected_option+1}.`}</p>{a.source_start != null && <button className="link-button" onClick={() => seek(a.source_start)}>Open {time(a.source_start)} <ArrowUpRight size={13} strokeWidth={1.7} aria-hidden="true"/></button>}</div></div>)}</div></div><div className="panel chart-panel"><div className="panel-header"><h2>Agent action counts</h2><Badge tone="amber">Simulation only</Badge></div><ResponsiveContainer width="100%" height={190}><BarChart data={activity} margin={{top:8,right:8,bottom:0,left:-28}}><CartesianGrid stroke="#e8ecec" vertical={false}/><XAxis dataKey="name" tick={{fontSize:11, fill:'#687679'}}/><YAxis tick={{fontSize:11, fill:'#687679'}}/><Tooltip/><Bar dataKey="count" fill="#507e79" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div></div></div>}
        {!runs.length && <div className="empty compact"><div className="empty-icon"><Globe2 size={22} strokeWidth={1.6} aria-hidden="true"/></div><h2>No trajectories yet</h2><p>Run the default profiles to create inspectable, constrained action traces.</p></div>}</> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Findings' && (lesson ? <><div className="page-head"><div><h1>Findings & repairs</h1><p>Each candidate weakness points to supplied source material and carries an evidence label.</p></div><Badge tone="amber">{findings.length} hypotheses</Badge></div><WorkspaceTabs view={view} onChange={setView}/>
        {findings.length ? <div className="finding-list">{findings.map(f => <article className="finding-card" key={f.id}><div className="finding-top"><Badge tone="amber">{f.evidence_label}</Badge><button className="source-link" onClick={() => seek(f.source_start)}>{time(f.source_start)} · Open passage <ArrowUpRight size={13} strokeWidth={1.7} aria-hidden="true"/></button></div><h2>{f.title}</h2><p>{f.explanation}</p><blockquote>{f.source_excerpt}</blockquote><div className="repair"><div><span className="eyebrow">PROPOSED REPAIR</span><p>{f.suggested_repair}</p></div><button disabled={busy} className="button secondary" onClick={() => makeRevision(f.id)}>Create revision <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button></div></article>)}</div> : <div className="empty compact"><div className="empty-icon"><Search size={22} strokeWidth={1.6} aria-hidden="true"/></div><h2>No findings yet</h2><p>Run a simulation to generate source-linked content checks.</p></div>}</> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}

      {view === 'Evidence' && (lesson ? <><div className="page-head"><div><h1>What the results support</h1><p>Model comparisons, dataset coverage and limitations stay separate from agent hypotheses.</p></div><a className="button secondary" href={`/api/reports/${lesson.id}/export`} target="_blank" rel="noreferrer">Export HTML report <ArrowUpRight size={13} strokeWidth={1.7} aria-hidden="true"/></a></div><WorkspaceTabs view={view} onChange={setView}/>
        <div className="evidence-status"><div className="evidence-mark">!</div><div><div className="eyebrow">CURRENT EVIDENCE STATUS</div><h2>Behavioral prediction unavailable</h2><p>{report.prediction?.reason || 'No validated benchmark artifact is attached to this lesson.'}</p></div></div>
        {research && <div className="readiness"><span className="eyebrow">RESEARCH EXECUTION GATE</span><div><Badge tone={research.numerical_execution_allowed ? 'green' : 'amber'}>{research.numerical_execution_allowed ? 'Precheck passed' : 'Blocked'}</Badge><span>Research volume {research.research_volume_mounted ? 'mounted' : 'unmounted'} · internal free {(research.internal_free_bytes / 1024**3).toFixed(1)} GiB / {(research.internal_reserve_bytes / 1024**3).toFixed(0)} GiB reserve. A separate label and media audit is still required.</span></div></div>}
        {parentLesson && <div className="panel comparison"><div className="panel-header"><h2>Version comparison</h2><Badge tone="amber">Content hypothesis only</Badge></div><div className="comparison-cols"><div><span>ORIGINAL · V{parentLesson.version}</span><strong>{parentLesson.sections.length} sections</strong><small>{parentRuns.length} scripted trajectories</small></div><div><span>CURRENT · V{lesson.version}</span><strong>{lesson.sections.length} sections</strong><small>{runs.length} scripted trajectories</small></div></div><p className="footnote">Matched profile and seed runs can be inspected in Simulation. Action counts are generated traces, not improvement scores. Equivalent human assessment has not been conducted.</p></div>}
        <div className="evidence-grid"><div className="panel"><div className="panel-header"><h2>Predetermined comparisons</h2><Badge tone="amber">Pending</Badge></div><table><thead><tr><th>Feature block</th><th>Attention MAE</th><th>Quiz Brier</th></tr></thead><tbody>{['Training mean / progress','Content only','Content + predicted EEG','Content + TRIBE','All blocks'].map(name => <tr key={name}><td>{name}</td><td>—</td><td>—</td></tr>)}</tbody></table><p className="footnote">A dash means no score was computed. It does not indicate zero error.</p></div><div className="panel"><div className="panel-header"><h2>Claim limits</h2></div><ul className="limit-list">{(report.limitations || ['No report is available yet.']).map(x => <li key={x}>{x}</li>)}</ul><div className="brain-placeholder"><div className="brain-symbol"><Brain size={29} strokeWidth={1.5} aria-hidden="true"/></div><div><strong>Predicted brain view pending</strong><span>Display only when a qualified TRIBE artifact exists. Never label it measured attention.</span></div></div></div></div>
        <div className="panel provenance"><div className="panel-header"><h2>Reproducibility record</h2></div><dl><div><dt>Lesson version</dt><dd>{lesson.version}</dd></div><div><dt>Source hash</dt><dd className="mono">{lesson.source_hash}</dd></div><div><dt>Content hash</dt><dd className="mono">{lesson.content_hash}</dd></div><div><dt>Analysis</dt><dd>{lesson.analysis_provenance}</dd></div><div><dt>Dataset / split</dt><dd>Not attached to this lesson</dd></div><div><dt>Human sample size</dt><dd>Not inferred from {runs.length} simulation runs</dd></div></dl></div></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}
      {view === 'Test' && (lesson ? <><div className="page-head"><div><h1>Brain visualization lab</h1><p>Explore an illustrative 3D brain and inspect local MRI and EEG files.</p></div><Badge tone="blue">Prototype</Badge></div><WorkspaceTabs view={view} onChange={setView}/><Suspense fallback={<div className="panel">Preparing 3D preview…</div>}><BrainTest/></Suspense></> : <EmptySelect onAdd={() => setView('NewLesson')}/>)}
      </div>{view !== 'NewLesson' && <aside className="inspector" aria-label="Workspace details"><section className="inspector-card"><div className="inspector-heading">Properties</div><dl><div><dt>Status</dt><dd><Badge tone={lesson?.status === 'ready' ? 'green' : 'neutral'}>{view === 'Projects' ? 'Local' : lesson?.status.replace('_', ' ') || 'Select lesson'}</Badge></dd></div><div><dt>Lesson version</dt><dd>{lesson && view !== 'Projects' ? `V${lesson.version}` : '—'}</dd></div><div><dt>Sections</dt><dd>{lesson && view !== 'Projects' ? lesson.sections.length : '—'}</dd></div><div><dt>Trajectories</dt><dd>{lesson && view !== 'Projects' ? runs.length : projects.reduce((n, p) => n + (p.trajectory_count || 0), 0)}</dd></div></dl></section><section className="inspector-card"><div className="inspector-heading">Source &amp; evidence</div>{lesson && view !== 'Projects' ? <><div className="inspector-source"><span className="source-icon"><NavIcon view="Lesson"/></span><span>{lesson.title}</span></div><p>{lesson.assessment.length} assessment {lesson.assessment.length === 1 ? 'item' : 'items'} · {lesson.transcript.length} transcript {lesson.transcript.length === 1 ? 'passage' : 'passages'}</p><button className="inspector-link" onClick={() => setView('Evidence')}>View evidence record <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button></> : <p>Select a lesson to see its source and evidence record.</p>}</section></aside>}</div>
    </main>
  </div>;
}

function EmptySelect({ onAdd }: { onAdd: () => void }) { return <div className="empty"><div className="empty-icon"><FileText size={22} strokeWidth={1.6} aria-hidden="true"/></div><h2>Select a lesson to continue</h2><p>Open an existing project or add a short video lesson.</p><button className="button primary" onClick={onAdd}>Add lesson</button></div>; }
