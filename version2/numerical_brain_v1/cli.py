"""Process one authenticated source with the versioned numerical backend."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'request-json', 'output', 'composite-model'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--request-id', required=True, help='Unique ID for existing shared-budget reservations')
    for name in ('cached-content-root', 'cached-audio-native-root', 'cached-presentation-root', 'speech-cache'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    from brain.worker_contract import PublicMediaRequest
    from .pipeline import run
    request = PublicMediaRequest.model_validate_json(a.request_json.read_text())
    result = run(a.source.resolve(), request, a.output.resolve(), a.request_id,
                 composite_model=a.composite_model.resolve(),
                 cached_content_root=a.cached_content_root,
                 cached_audio_native_root=a.cached_audio_native_root,
                 cached_presentation_root=a.cached_presentation_root,
                 speech_cache=a.speech_cache)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
