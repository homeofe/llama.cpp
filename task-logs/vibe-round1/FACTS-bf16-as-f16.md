# FACTS: GGML_CUDA_BF16_AS_F16 (Turing, Qwen3.8-Flash-Next)

Für den Eigentümer, alle Zahlen mit Quelle. Stand: 2026-10-19, Worktree `llama-vibe`, Branch `vibe-parity`.

## Warum (Stage-1/2-Messungen, task-logs/vibe-round1/)

- Die Runde unseres Servers ist 81,7 ms (Code); Strata 47,2-47,8 ms. Die ganze Lücke pro Token
  (22,5 gegen 14,5 ms) sitzt im dichten kleinen-Batch-GPU-Pfad, nicht im Experten-Pfad
  (CPU-Experten, Uploads, Cache sind auf Augenhöhe; NOTES-vibe-round1.md).
- Die Hyper-Connection-Matrizen (hc_attn/hc_ffn up, je 10240x320, BF16) laufen auf der 2080 Ti im
  zeilenweisen MMVF-Kernel: 85,6 GB/s, 76,6 us pro Aufruf (`bound/series2.sh`, 3 Runden, ruhig).
  Dieselbe Shape in F16: 308 GB/s, 21,3 us - 3,6x, nur weil der F16-Dispatch auf Turing den
  Tensor-Core-Pfad (MMF) nimmt, BF16 aber im Vektor-Kernel hängen bleibt
  (`ggml_cuda_should_use_mmvq`, BF16-Zweig).
- Bound end-to-end mit der bestehenden IQ3_S-hcfix-Datei (hc up/Router F16, hc down Q8_0):
  `q-vibe-hcfix` gegen `q-vibe-control`, 3 Runden, je 12 saubere Anfragen pro Kategorie:
  Code 50,6 gegen 47,7 tok/s (+6,1 %), Englisch 42,2 gegen 39,3 (+7,4 %).

## Was gebaut wurde

`GGML_CUDA_BF16_AS_F16=1` (default aus, aus = unverändert bit für bit):

- bf16_f16.cu/.cuh: In-place-Konvertierung bf16 -> f16 (gleiche Elementgröße, kein Extra-VRAM),
  einmal je Gewicht; Registry der konvertierten Zeiger; Graph-Survey, der NUR konvertiert, wenn
  jeder Knoten des Graphen den Tensor ausschließlich über MUL_MAT konsumiert (get_rows, mul, add,
  mul_mat_id usw. vetozieren; 3D-Experten niemals).
- ggml-cuda.cu: mul_mat reinterpretiert registrierte BF16-src0 als F16, der F16-Dispatch (MMF,
  Tensor Cores) wählt dann den schnellen Kern. Konvertierung läuft im Warmup (nie in ein CUDA-
  Graph-Capture).
- Kernel bewusst als Integer-Arithmetik: die Hardware-cvt bzw. die fusionierte bf16->f16-Kette
  flusht f16-Subnormale (gemessen: 2^-24 -> 0 statt 0x0001); die Integer-Version rundet
  IEEE-round-to-nearest-even, bewiesen gegen F16C auf dem Host.

## Beweise

1. `bound/check_bf16_f16`: 1542 Bitmuster (jeder Exponent, Subnormale, Overflow, NaN) gegen eine
   unabhängige Software-Referenz (die gegen F16C verifiziert ist): 0 Abweichungen, PASS.
2. Mutation-Cuts: Sign-Bit im Subnormal-Zweig entfernt -> 339 Abweichungen (FAIL);
   1-ulp-Tie-Rundung geändert -> 616 Abweichungen (FAIL); restauriert -> 0, PASS.
3. `bound/check_mul_mat_bf16`: 4 hc-Shapes, CPU-Referenz gegen CUDA: nmse 0 (bitweise gleich),
   Schalter aus UND ein.
4. Geschwindigkeit (ruhig, 3 Runden): hc_up 76,7 -> 20,5 us (85,4 -> 320 GB/s, 3,7x).
5. End-to-End: stage_b 3 Runden, je 12 saubere Anfragen pro Konfiguration und Kategorie
   (`vibe-patch-e2e/`): Kontrolle 47,7 Code / 39,4 Englisch tok/s; Patch 48,8 (+2,3 %) /
   40,8 (+3,6 %); Rundenzeit laut Server-Uhr 79,0/75,9 -> 77,8/74,5 ms. Unter dem deklarierten
   Tor von +3 % auf BEIDEN Kategorien (Code fehlt). Der Kernel-Gewinn (5,3 ms/Runde) läuft
   großteils im Schatten der CPU-Experten-Phase; an der Wand bleiben nur 1,2-1,4 ms.
   => NICHT als alleinige Änderung übernommen.
6. Perplexität test-de, 12 Chunks, ctx 512: -ub 4: 3,7963 -> 3,7990 (+0,07 %); -ub 512:
   3,8200 -> 3,7949 (-0,66 %). Beides innerhalb der +0,5 %-Grenze; die Kontrolle trifft den
   historischen Wert 3,7963 exakt (Schalter aus = unverändert).
7. Vergleich Datei-Retype im selben Protokoll (IQ3_S-hcfix: hc up/Router F16 UND hc down Q8_0):
   Code +6,1 %, Englisch +7,4 % - ÜBER dem Tor. Empfehlung: hcfix-Datei für iq3_s ausrollen
   (keine Code-Änderung nötig); Runtime-Patch als bewiesene Grundlage behalten (hilft jedem
   BF16-dichten Modell auf dieser GPU, und beim Qwen3.8 in Kombination mit weiteren
   Kernel-Schritten, die die exponierte Zeitlinie angreifen: die ~14 ms/Runde
   Elementwise-/Norm-Kette in ~4.700 Kerneln pro Runde).

## Offen / Empfehlung

- Deployment: der Patch ist default-aus; Empfehlung, ihn für `iq3_s` auf dem Turing-Runtime zu
  aktivieren, NACHDEM E2E und PPL durch sind. Keine Live-Einstellung wurde geändert.
- Der Rest der Dense-Lücke (~20 ms/Runde) bleibt: Elementwise-/Norm-Kette fusionieren (~14 ms),
  MMVQ-Batching für Q6_K/Q4_K (~5 ms), gruppierte Hit-Kernel (~3,5 ms) - eigenes Kernel-Paket.

## Nachtrag: MMVQ-Turing-Tabelle + Paket-Ergebnis (2026-10-19, Abend)

- Die Tabelle `calc_nwarps` (Turing, ncols_dst 2-4) stand auf 4; der Sweep ueber 6 Konfigurationen
  auf den echten Shapes (Q6_K 10240x2560 / 2560x6144, Q4_K 6144x2560, n=4/5) waehlt nwarps=2.
  Unter CPU-Last (15-22) gewinnt die getunte Konfiguration 30-60% (attn_qkv 300-341 gegen
  182-207 GB/s), ruhig ~5-7%. Korrektheit: 2.444 tbo-MUL_MAT-Faelle OK (identisch zur Baseline).
- E2E (stage_b, 3 Runden interleaved, n=12, 0 Requeues, Fremdlast 7-11):
  Kontrolle 45,7/37,3; nur Tabelle 47,7/38,5 (+4,4/+3,2%); Tabelle+bf16-als-F16 48,9/40,5
  (+7,0/+8,6%). Beide ueber dem +3%-Tor; Empfehlung: das Paket.
- Perplexitaet der Tabellenaenderung: siehe NOTES (Abschnitt oben); Grenze +0,5%.
