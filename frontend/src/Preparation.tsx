import { ArrowRight, BookOpenText, Brain, CircleCheck, ClipboardList, FileVideo, Globe2, Plus, Search, Trash2, Users } from 'lucide-react';
import type { Item, Lesson, Profile } from './types';
import './preparation.css';

export type PreparationDestination = 'Lesson' | 'Questions' | 'Learners' | 'Simulation' | 'Findings' | 'Evidence';

const iconProps = { size: 17, strokeWidth: 1.7, 'aria-hidden': true as const };

export function LessonOverview({ lesson, questionCount, profileCount, runCount, findingCount, onNavigate }: {
  lesson: Lesson;
  questionCount: number;
  profileCount: number;
  runCount: number;
  findingCount: number;
  onNavigate: (view: PreparationDestination) => void;
}) {
  const next: { view: PreparationDestination; title: string; description: string } = lesson.status !== 'ready'
    ? { view: 'Lesson', title: 'Finish the source transcript', description: 'A timed transcript connects the lesson to source-linked findings.' }
    : !lesson.concepts_reviewed
      ? { view: 'Lesson', title: 'Review the extracted concepts', description: 'Confirm the concepts before you interpret the learner traces.' }
      : !questionCount
        ? { view: 'Questions', title: 'Add an assessment question', description: 'Give the simulation a concrete outcome to inspect.' }
        : !profileCount
          ? { view: 'Learners', title: 'Set up learner profiles', description: 'Choose prior knowledge and misconceptions for the simulated learners.' }
          : !runCount
            ? { view: 'Learners', title: 'Run the first simulation', description: 'Review the profiles and run limits before generating trajectories.' }
            : { view: findingCount ? 'Findings' : 'Simulation', title: findingCount ? 'Inspect the findings' : 'Inspect the trajectories', description: findingCount ? 'Each proposed repair is linked to a passage in the source.' : 'Look through the learner actions before drawing conclusions.' };

  const sections = [
    { view: 'Lesson' as const, title: 'Source', description: 'Video, timed transcript, sections and concepts', status: lesson.status === 'ready' ? (lesson.concepts_reviewed ? 'Reviewed' : 'Review concepts') : 'Needs transcript', icon: <FileVideo {...iconProps}/> },
    { view: 'Questions' as const, title: 'Questions', description: 'Assessment prompts, choices and answer keys', status: `${questionCount} ${questionCount === 1 ? 'question' : 'questions'}`, icon: <ClipboardList {...iconProps}/> },
    { view: 'Learners' as const, title: 'Learners', description: 'Prior knowledge, misconceptions and run limits', status: `${profileCount} ${profileCount === 1 ? 'profile' : 'profiles'}`, icon: <Users {...iconProps}/> },
    { view: 'Simulation' as const, title: 'Runs', description: 'Seeded learner actions and inspectable traces', status: `${runCount} trajectories`, icon: <Globe2 {...iconProps}/> },
    { view: 'Findings' as const, title: 'Findings', description: 'Source-linked difficulties and proposed repairs', status: `${findingCount} hypotheses`, icon: <Search {...iconProps}/> },
    { view: 'Evidence' as const, title: 'Evidence', description: 'Comparison limits and reproducibility record', status: 'Human validation pending', icon: <Brain {...iconProps}/> },
  ];

  return <div className="prep-overview">
    <section className="prep-next panel">
      <div className="prep-next-icon"><BookOpenText size={20} strokeWidth={1.6} aria-hidden="true"/></div>
      <div><span className="eyebrow">SUGGESTED NEXT STEP</span><h2>{next.title}</h2><p>{next.description}</p></div>
      <button className="button primary" onClick={() => onNavigate(next.view)}>Open {next.view === 'Lesson' ? 'Source' : next.view === 'Simulation' ? 'Runs' : next.view} <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button>
    </section>
    <div className="prep-overview-metrics" aria-label="Lesson summary">
      <div><span>SECTIONS</span><strong>{lesson.sections.length}</strong><small>From the supplied source</small></div>
      <div><span>QUESTIONS</span><strong>{questionCount}</strong><small>Educator answer key</small></div>
      <div><span>TRAJECTORIES</span><strong>{runCount}</strong><small>Simulated, not human</small></div>
    </div>
    <div className="section-heading"><h2>Workspace</h2><span>Choose a section to continue</span></div>
    <div className="prep-section-grid">{sections.map(section => <button key={section.view} className="prep-section-card" onClick={() => onNavigate(section.view)}>
      <span className="prep-section-icon">{section.icon}</span><span className="prep-section-title">{section.title}</span><span className="prep-section-description">{section.description}</span><span className="prep-section-status">{section.status}</span><ArrowRight className="prep-section-arrow" size={15} strokeWidth={1.7} aria-hidden="true"/>
    </button>)}</div>
  </div>;
}

export function QuestionEditor({ items, onChange, onSave, busy }: {
  items: Item[];
  onChange: (items: Item[]) => void;
  onSave: () => void;
  busy: boolean;
}) {
  const valid = items.length > 0 && items.every(item => item.prompt.trim() && item.options.length >= 2 && item.options.length <= 6 && item.options.every(option => option.trim()) && Number.isInteger(item.correct_index) && item.correct_index! >= 0 && item.correct_index! < item.options.length);
  const update = (id: string, patch: Partial<Item>) => onChange(items.map(item => item.id === id ? { ...item, ...patch } : item));
  const addQuestion = () => onChange([...items, { id: `q-${crypto.randomUUID()}`, prompt: '', options: ['', ''], correct_index: 0 }]);
  const removeOption = (item: Item, optionIndex: number) => {
    if (item.options.length <= 2) return;
    const options = item.options.filter((_, index) => index !== optionIndex);
    const answer = item.correct_index ?? 0;
    update(item.id, { options, correct_index: answer > optionIndex ? answer - 1 : Math.min(answer, options.length - 1) });
  };

  return <div className="prep-questions">
    <div className="prep-note"><CircleCheck size={16} strokeWidth={1.7} aria-hidden="true"/><span>Answer keys are stored outside student action records. Saving creates a new lesson version.</span></div>
    <div className="prep-question-list">{items.map((item, index) => <section className="panel prep-question-card" key={item.id}>
      <div className="panel-header"><div><span className="prep-index">Q{index + 1}</span><h2>Question {index + 1}</h2></div><button className="prep-icon-action" type="button" title="Remove question" aria-label={`Remove question ${index + 1}`} disabled={items.length <= 1} onClick={() => onChange(items.filter(other => other.id !== item.id))}><Trash2 size={15} strokeWidth={1.7} aria-hidden="true"/></button></div>
      <label className="prep-field">Prompt<input value={item.prompt} onChange={event => update(item.id, { prompt: event.target.value })} placeholder="What should the learner understand?"/></label>
      <div className="prep-options"><div className="prep-field-heading"><span>Answer options</span><small>Choose the correct answer below</small></div>{item.options.map((option, optionIndex) => <div className="prep-option-row" key={optionIndex}><span className="prep-option-letter">{String.fromCharCode(65 + optionIndex)}</span><input aria-label={`Question ${index + 1} option ${optionIndex + 1}`} value={option} onChange={event => update(item.id, { options: item.options.map((value, i) => i === optionIndex ? event.target.value : value) })} placeholder={`Option ${optionIndex + 1}`}/><button className="prep-icon-action" type="button" title="Remove option" aria-label={`Remove option ${optionIndex + 1} from question ${index + 1}`} disabled={item.options.length <= 2} onClick={() => removeOption(item, optionIndex)}><Trash2 size={14} strokeWidth={1.7} aria-hidden="true"/></button></div>)}</div>
      <div className="prep-question-footer"><label className="prep-field">Correct answer<select value={item.correct_index ?? 0} onChange={event => update(item.id, { correct_index: Number(event.target.value) })}>{item.options.map((_, optionIndex) => <option key={optionIndex} value={optionIndex}>Option {String.fromCharCode(65 + optionIndex)}</option>)}</select></label><button type="button" className="button secondary" disabled={item.options.length >= 6} onClick={() => update(item.id, { options: [...item.options, ''] })}><Plus size={13} strokeWidth={1.7} aria-hidden="true"/>Add option</button></div>
    </section>)}</div>
    <div className="prep-actions"><button className="button secondary" type="button" onClick={addQuestion}><Plus size={13} strokeWidth={1.7} aria-hidden="true"/>Add question</button><button className="button primary" type="button" disabled={busy || !valid} onClick={onSave}>Save questions</button></div>
    {!valid && <p className="prep-validation">Enter a prompt and at least two filled options for every question before saving.</p>}
  </div>;
}

export type RunSettings = { profileCount: number; seeds: number; maxActions: number; maxTokens: number; maxCost: number };

export function LearnerSetup({ profiles, onProfilesChange, settings, onSettingsChange, onSave, onRun, busy, saved, lessonReady }: {
  profiles: Profile[];
  onProfilesChange: (profiles: Profile[]) => void;
  settings: RunSettings;
  onSettingsChange: (patch: Partial<RunSettings>) => void;
  onSave: () => void;
  onRun: () => void;
  busy: boolean;
  saved: boolean;
  lessonReady: boolean;
}) {
  const total = settings.profileCount * settings.seeds;
  const valid = profiles.length >= 1 && profiles.length <= 12 && profiles.every(profile => profile.name.trim()) && Number.isInteger(settings.profileCount) && settings.profileCount >= 1 && settings.profileCount <= profiles.length && Number.isInteger(settings.seeds) && settings.seeds >= 1 && settings.seeds <= 100 && total <= 100 && Number.isInteger(settings.maxActions) && settings.maxActions >= 1 && settings.maxActions <= 100 && Number.isInteger(settings.maxTokens) && settings.maxTokens >= 0 && settings.maxTokens <= 10000 && Number.isFinite(settings.maxCost) && settings.maxCost >= 0 && settings.maxCost <= 5;
  const updateProfile = (id: string, patch: Partial<Profile>) => onProfilesChange(profiles.map(profile => profile.id === id ? { ...profile, ...patch, provenance: 'educator-authored' } : profile));
  const removeProfile = (id: string) => {
    if (profiles.length <= 1) return;
    onProfilesChange(profiles.filter(profile => profile.id !== id));
    if (settings.profileCount >= profiles.length) onSettingsChange({ profileCount: profiles.length - 1 });
  };
  const addProfile = () => onProfilesChange([...profiles, { id: `p-${crypto.randomUUID()}`, name: `Learner ${profiles.length + 1}`, prior_knowledge: 'medium', misconceptions: [], provenance: 'educator-authored' }]);

  return <div className="prep-learners">
    <div className="prep-note"><Users size={16} strokeWidth={1.7} aria-hidden="true"/><span>These are authored learner profiles. Their trajectories are simulations, not observations of real students.</span></div>
    <div className="prep-learners-grid"><section className="panel prep-profile-panel"><div className="panel-header"><div><h2>Learner profiles</h2><p>{profiles.length} available · up to 12 per lesson</p></div><button className="button secondary" type="button" disabled={profiles.length >= 12} onClick={addProfile}><Plus size={13} strokeWidth={1.7} aria-hidden="true"/>Add profile</button></div>
      <div className="prep-profile-list">{profiles.map((profile, index) => <div className="prep-profile-row" key={profile.id}><span className="prep-index">{String(index + 1).padStart(2, '0')}</span><div className="prep-profile-fields"><label className="prep-field">Name<input aria-label={`Profile ${index + 1} name`} value={profile.name} onChange={event => updateProfile(profile.id, { name: event.target.value })}/></label><label className="prep-field">Prior knowledge<select aria-label={`Profile ${index + 1} prior knowledge`} value={profile.prior_knowledge} onChange={event => updateProfile(profile.id, { prior_knowledge: event.target.value as Profile['prior_knowledge'] })}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label><label className="prep-field prep-misconceptions">Misconceptions<input aria-label={`Profile ${index + 1} misconceptions`} value={profile.misconceptions.join(', ')} onChange={event => updateProfile(profile.id, { misconceptions: event.target.value.split(',').map(value => value.trim()).filter(Boolean) })} placeholder="Comma-separated concepts"/></label></div><button className="prep-icon-action" type="button" title="Remove profile" aria-label={`Remove profile ${index + 1}`} disabled={profiles.length <= 1} onClick={() => removeProfile(profile.id)}><Trash2 size={14} strokeWidth={1.7} aria-hidden="true"/></button></div>)}</div>
      <div className="prep-profile-actions"><button className="button secondary" type="button" disabled={busy || profiles.length < 1 || profiles.some(profile => !profile.name.trim())} onClick={onSave}>Save profiles</button><small>{saved ? 'Profiles saved to this lesson.' : 'Saved profiles can be reused in later runs of this lesson.'}</small></div>
    </section><aside className="panel prep-run-panel"><div className="panel-header"><h2>Run setup</h2><span>{Number.isFinite(total) ? total : 0} trajectories</span></div><p>Choose how many profiles to simulate and the bounds for each run.</p>
      <div className="prep-settings"><label className="prep-field">Profiles to run<input type="number" min="1" max={profiles.length} value={settings.profileCount} onChange={event => onSettingsChange({ profileCount: Number(event.target.value) })}/></label><label className="prep-field">Seeds per profile<input type="number" min="1" max="100" value={settings.seeds} onChange={event => onSettingsChange({ seeds: Number(event.target.value) })}/></label><label className="prep-field">Max actions per run<input type="number" min="1" max="100" value={settings.maxActions} onChange={event => onSettingsChange({ maxActions: Number(event.target.value) })}/></label><label className="prep-field">Max tokens per run<input type="number" min="0" max="10000" value={settings.maxTokens} onChange={event => onSettingsChange({ maxTokens: Number(event.target.value) })}/></label><label className="prep-field">Cost ceiling per run ($)<input type="number" min="0" max="5" step="0.1" value={settings.maxCost} onChange={event => onSettingsChange({ maxCost: Number(event.target.value) })}/></label></div>
      <p className="prep-run-limit">Maximum 100 trajectories. The current scripted renderer uses zero tokens and has no model cost.</p><button className="button primary full" disabled={busy || !valid || !lessonReady} onClick={onRun}>Run {Number.isFinite(total) ? total : 0} trajectories <ArrowRight size={13} strokeWidth={1.7} aria-hidden="true"/></button>{!lessonReady && <small className="prep-validation">Add a transcript before running.</small>}{!valid && <small className="prep-validation">Check the profile names and run limits.</small>}
    </aside></div>
  </div>;
}
