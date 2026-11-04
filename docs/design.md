# Interface direction

Perception is a lesson simulation workspace. Teachers and curriculum reviewers open a lesson from the library, review its overview and source, edit questions, choose learner profiles, run constrained trajectories, and inspect findings and evidence. Lab is a separate visualization workbench within each lesson.

The root page is a brief introduction with one primary action into the workspace. It shows a real synthetic-project screenshot, states the three core steps, and labels simulated findings as hypotheses. The sidebar brand returns to the introduction; `#workspace` opens the app directly.

## Reference lock

The three user-provided Aaru screenshots from 2026-11-01 are the visual reference. They show a white application surface with a narrow left navigation rail, page-specific pill tabs, rounded hairline cards, and a right properties rail. Perception keeps its pages flush to the top without a separate breadcrumb bar. The landscape outside Aaru's app and its retail-specific questions, audiences, files, and sharing controls do not carry over.

- **Canvas and typography:** white throughout; near-black headings, neutral gray supporting text, Geist Sans for the interface, Geist Mono for the Perception wordmark and technical identifiers, generous whitespace.
- **Layout:** persistent recent-lessons sidebar; lesson title and tabs in the center; contextual status and source details in the right inspector. The inspector moves below content on narrow screens.
- **Controls:** neutral pills and subtle borders; blue reserved for primary actions and focus. Green marks ready status, amber marks authored assumptions and unvalidated evidence.
- **Sidebar states:** the darker selection background slides to the chosen row when navigation changes; the lighter hover and keyboard focus background stays fixed to each row. The brand row sits close to the top edge, and a compact control collapses the sidebar to icons.
- **Icons:** use one restrained, thin-stroke icon family throughout the browser interface. Lucide supplies the web-safe equivalents to the requested SF Symbols look; the custom Perception mark remains the brand icon.
- **Content:** video, transcript, assessment, learner traces, findings, and evidence labels keep their existing behavior. Simulation output is never presented as human validation.
- **Add lesson:** the upload, transcript, and assessment form opens in the main workspace. It has source and assessment sections, with a direct return to the Lessons library.
- **Identity:** use the supplied `Union (1).svg` as the Perception symbol. Keep it as an SVG, paired with a Geist Mono wordmark in the sidebar and used as the browser favicon.
- **Scale:** buttons, tabs, cards, and metadata use a denser application scale while preserving readable labels and click targets.
- **Test:** a dark viewport contains a rotatable anatomical preview with relative EEG color overlays. Local NIfTI slices appear in a separate preview and as an unregistered plane in the 3D scene. No scan or signal is sent to the backend. The bundled anatomical mesh is [Innerscene's CC0 Brain](https://www.innerscene.com/tools/library/3d-parts/brain-10d8f88f), adapted from MakeHuman CC0 assets; it is illustrative anatomy rather than this lesson's MRI-derived surface.

Refero style search was attempted earlier but returned `NO_SUBSCRIPTION`; the attached screenshots provide the direct style target.
