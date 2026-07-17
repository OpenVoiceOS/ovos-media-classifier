# Changelog

## [0.0.1a1](https://github.com/OpenVoiceOS/ovos-media-classifier/tree/0.0.1a1) (2026-07-17)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-media-classifier/compare/30cdc86a580c5bfbbe199e501861ed75ffe6649d...0.0.1a1)

**Merged pull requests:**

- fix: install cleanly in a fresh environment [\#47](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/47) ([JarbasAl](https://github.com/JarbasAl))
- docs: mark as pre-release work-in-progress [\#46](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/46) ([JarbasAl](https://github.com/JarbasAl))
- docs: timeless README/docs cleanup pass [\#38](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/38) ([JarbasAl](https://github.com/JarbasAl))
- feat: final consolidated rebuild + retrain with ASR-noise realism layer [\#37](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/37) ([JarbasAl](https://github.com/JarbasAl))
- feat: context-aware classify\_full + gazetteer ambiguity-abstain + resolved\_rate [\#36](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/36) ([JarbasAl](https://github.com/JarbasAl))
- feat\(locale\): agentpipe-generated conversational slot templates [\#35](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/35) ([JarbasAl](https://github.com/JarbasAl))
- feat: metadatarr-backed routing layers for the open-vocab gap [\#34](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/34) ([JarbasAl](https://github.com/JarbasAl))
- feat\(locale\): conversational/spoken-register templates [\#33](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/33) ([JarbasAl](https://github.com/JarbasAl))
- feat: guided-categorical-embeddings embedding-router backend [\#32](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/32) ([JarbasAl](https://github.com/JarbasAl))
- fix: close routing harms surfaced by the harm-weighted eval [\#31](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/31) ([JarbasAl](https://github.com/JarbasAl))
- feat\(benchmarks\): harm-weighted out-of-distribution routing eval [\#30](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/30) ([JarbasAl](https://github.com/JarbasAl))
- feat: emit PictureFormat + import Structure from mediavocab \(convergence Phase 2\) [\#29](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/29) ([JarbasAl](https://github.com/JarbasAl))
- feat!: emit mediavocab axes instead of flat qualifiers/tags \(convergence Phase 1\) [\#28](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/28) ([JarbasAl](https://github.com/JarbasAl))
- feat: trailer/BTS/supplementary content as qualifiers [\#27](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/27) ([JarbasAl](https://github.com/JarbasAl))
- feat\(locale\): saturate templates + confusable matrix [\#26](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/26) ([JarbasAl](https://github.com/JarbasAl))
- feat\(locale\): expand keyword-entity-under-unexpected-label confusables [\#25](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/25) ([JarbasAl](https://github.com/JarbasAl))
- docs: link published HF model collection + download [\#24](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/24) ([JarbasAl](https://github.com/JarbasAl))
- feat: neural-net classifiers + richer text features + full ladder comparison [\#23](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/23) ([JarbasAl](https://github.com/JarbasAl))
- feat\(training\): hierarchical coarse-to-fine classifier experiment [\#22](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/22) ([JarbasAl](https://github.com/JarbasAl))
- feat\(training\): rebuild on saturated templates + UNIFIED entity sets [\#21](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/21) ([JarbasAl](https://github.com/JarbasAl))
- fix\(locale\): comprehensive keyword vocabularies for default backend [\#20](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/20) ([JarbasAl](https://github.com/JarbasAl))
- feat\(locale\): saturate sentence-template variation [\#19](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/19) ([JarbasAl](https://github.com/JarbasAl))
- refactor!: move dataset templates into translatable locale/ .intent+.voc, drop python generator [\#18](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/18) ([JarbasAl](https://github.com/JarbasAl))
- refactor: tidy runtime backends \(dead code, dup, stale docs\) [\#17](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/17) ([JarbasAl](https://github.com/JarbasAl))
- docs: 10/10 README + docs coherence sweep [\#16](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/16) ([JarbasAl](https://github.com/JarbasAl))
- feat: tags head \(genre/mood/era folded\), IMDb relational data, final rebuild [\#15](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/15) ([JarbasAl](https://github.com/JarbasAl))
- feat: multi-task per-axis ONNX classifier + sklearn ladder + per-axis benchmark + docs [\#14](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/14) ([JarbasAl](https://github.com/JarbasAl))
- feat\(training\): full media-metadata ingestion + standardized dataset generator [\#13](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/13) ([JarbasAl](https://github.com/JarbasAl))
- refactor!: drop OCPPlayIntent; classify on mediavocab MediaType + genres directly [\#12](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/12) ([JarbasAl](https://github.com/JarbasAl))
- feat\(training\): componential template generator \(~6k diverse templates\) [\#10](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/10) ([JarbasAl](https://github.com/JarbasAl))
- docs: polish README + docs to 10/10 for first publish [\#9](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/9) ([JarbasAl](https://github.com/JarbasAl))
- fix\(training\): drop agy/antigravity from agentpipe pool [\#8](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/8) ([JarbasAl](https://github.com/JarbasAl))
- refactor: load/match .voc via ovos-spec-tools \(word-boundary, global rule\) [\#7](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/7) ([JarbasAl](https://github.com/JarbasAl))
- feat: hierarchical coarse-to-fine keyword classification \(modality→structure→constrained leaf\) [\#6](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/6) ([JarbasAl](https://github.com/JarbasAl))
- feat: optional ONNX trained-classifier backend \(raw onnxruntime+numpy\) [\#5](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/5) ([JarbasAl](https://github.com/JarbasAl))
- docs: develop multi-axis classification theory + polish docs [\#4](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/4) ([JarbasAl](https://github.com/JarbasAl))
- docs: add runnable examples [\#3](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/3) ([JarbasAl](https://github.com/JarbasAl))
- feat: multi-axis classification \(Structure + playback\_type\) [\#2](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/2) ([JarbasAl](https://github.com/JarbasAl))
- chore: Configure Renovate [\#1](https://github.com/OpenVoiceOS/ovos-media-classifier/pull/1) ([renovate[bot]](https://github.com/apps/renovate))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
