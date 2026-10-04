# Third-party boundaries

This repository contains original implementation code. It does not vendor upstream research
implementations, checkpoints, commercial services, benchmark videos, or annotations. Compatibility
with public interfaces does not imply endorsement or an upstream benchmark reproduction.

Python packages are declared in `pyproject.toml`; their separate licenses apply. FFmpeg/PyAV build
configurations can affect codec/license obligations. This delivery is not a transitive legal audit.
Generate an SBOM and review dependency/model/data/service terms before distributing a product.

The optional `gameboy` extra installs PyBoy under its own terms. No emulator implementation,
commercial ROM, game asset, walkthrough or privileged-state game harness is vendored. The
interactive package was integrated from the owner's StreamBudget source delivery; its archive
hash, reviewed base and consolidation boundary are recorded in `docs/INTERACTIVE.md`. Supply
authorized GB/GBC ROMs privately and do not publish generated game observations or checkpoints
without reviewing their rights. The synthetic interactive fixture uses original toy pixels.

StreamArena's public repository describes evaluation code separately from its CC-BY-NC annotations;
underlying video rights are another layer. Its annotations are evaluation questions/reference
answers, not a deployment dependency. Use authorized footage and independently created labels.
Machine-generated captions/OCR on customer data are runtime outputs, not copied benchmark labels.

`tests/` and `demo.py` generate original synthetic media and answer keys solely for plumbing tests.
No claimed visual-model accuracy result is derived from those fixtures.

See `docs/REFERENCES.md` for the official Creative Commons terms and upstream source locations.

The optional `scripts/prepare_footage.py` fetches five openly licensed Wikimedia Commons
clips for exploratory QA. These files remain in ignored local data directories and
are not part of the repository's MIT-licensed software. Source-specific attribution,
license links, derivative descriptions and hashes are in
`research/REAL_SCREEN_20260920.md` and its JSON receipt. Downloading is explicit opt-in;
the script fails if the expected permissive rights metadata changes.
