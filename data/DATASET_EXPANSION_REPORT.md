# AttackLM Dataset Expansion Report
## Generated: 2026-06-09

---

## Executive Summary

Four parallel research/scraping agents investigated new data sources for AttackLM training data. Results:

| Source | Type | Est. Training Pairs | Priority |
|--------|------|---------------------|----------|
| **InternalAllTheThings** | Red team tactics website | 3,500–5,000 | 🔴 HIGH |
| **rgbwiki Red Cell** | Red team tactics wiki | 800–1,200 | 🟡 MEDIUM |
| **HackingArticles.in** | Pen-testing blog (414 articles) | 1,250–1,700 | 🔴 HIGH |
| **AI Red Team Tools (25)** | Prompt/jailbreak datasets | 33,000+ | 🔴 HIGH |
| **TOTAL NEW DATA** | | **~38,550–40,900 pairs** | |

---

## Part 1: Red Team Tactics Websites

### 1A. InternalAllTheThings (swisskyrepo.github.io)

**Site Structure:** 150+ pages across 10 major categories

| Category | Pages | Key Content |
|----------|-------|-------------|
| Active Directory | 60+ | Kerberoasting, AS-REP, NTLM relay, RBCD, ADCS (ESC1-ESC15), DCSync, Golden/Silver Tickets, Skeleton Key, BloodHound |
| Red Team | 40+ | Access, escalation, evasion, persistence, pivoting |
| Cloud | 35+ | AWS, Azure, IBM Cloud attacks |
| Cheatsheets | 10+ | Hash cracking, shells, Mimikatz, PowerShell |
| Command & Control | 5+ | Cobalt Strike, Metasploit, Mythic |
| Containers | 2+ | Docker, Kubernetes |
| Databases | 5+ | MSSQL attacks |
| DevOps | 8+ | CI/CD attacks, secrets enumeration |
| Methodology | 4+ | Bug hunting, source code analysis |
| CVE | 5+ | MS14-068, ZeroLogon, PrintNightmare |

**Top Techniques Found:**
- Kerberoasting (T1558.003) — 15+ variants
- AS-REP Roasting (T1558.004) — 8+ variants
- NTLM Relay (T1557) — 13 techniques including CVE-2025-33073
- ADCS ESC Attacks (T1649) — 15 variants (ESC1-ESC15)
- DCSync/NTDS Dumping (T1003.006) — 10+ variants
- Pass-the-Hash (T1550.002) — 20+ variants
- Windows Privilege Escalation — 50+ techniques
- Windows Persistence — 30+ mechanisms
- Token Impersonation (T1134.001) — JuicyPotato, RoguePotato, PrintSpoofer
- WMI Event Subscription (T1546.003), Scheduled Tasks (T1053.005), Winlogon Helper DLL (T1547.004)

**Estimated Training Pairs:** 3,500–5,000
- AD Attacks: 1,500–2,000
- Privilege Escalation: 800–1,200
- Persistence: 500–700
- Credential Access: 400–600
- Defense Evasion: 300–500
- Lateral Movement: 200–300

**MITRE Coverage:** 15+ technique IDs mapped with concrete commands

---

### 1B. rgbwiki Red Cell (rgbwiki.com/Red Cell)

**Site Structure:** 15 major sections, 60+ sub-pages

| Section | Techniques | Status |
|---------|------------|--------|
| Active Directory | 8+ (Kerberoasting, AS-REP, Golden Ticket, Delegation) | ✅ Full |
| Wireless (802.11) | 6+ (WEP/WPA2/WPS cracking) | ✅ Full |
| Enumeration | 8+ (Network, service, web, AD) | ✅ Full |
| Privilege Escalation | 13+ (Windows + Linux vectors) | ✅ Full |
| Tunneling & Lateral Movement | 8+ (PsExec, RDP, SSH, Chisel) | ✅ Full |
| File Transfers | 3+ | ✅ Full |
| Payloads | 8+ (Reverse shells, webshells, macros) | ✅ Full |
| Defense Evasion | 4+ (AMSI bypass, Defender disabling) | ✅ Full |
| Credential Dumping | 5+ (Mimikatz, LSASS, SAM) | ✅ Full |
| Persistence | 3+ (Registry, scheduled tasks, startup) | ✅ Full |
| Web Exploitation | 7+ (XSS, SSRF, XXE, SSTI, IDOR, CSRF) | ✅ Full |
| SQL | 3+ (MSSQL, MySQL, SQLi) | ✅ Full |
| ICS | 3+ (Allen-Bradley, Siemens) | ⚠️ WIP |
| Cheatsheets | 7+ (Impacket, Rubeus, MSFvenom, C2) | ⚠️ WIP |
| Technique Ted Talks | 3+ (Binary exploitation, RF, .NET) | ⚠️ WIP |

**Estimated Training Pairs:** 800–1,200

**Limitations:** Several pages marked WIP (AD CS, Delegation exploitation, DLL Hijacking, C2 cheatsheets). Some sections lack detailed command examples.

---

### 1C. HackingArticles.in (Penetration Testing Category)

**Scope:** 46 category pages, ~414 total articles estimated

**Articles Scraped:** 8 high-value technical articles fully analyzed

**Top Articles Found:**
1. UAC Bypass — 8 distinct bypass techniques with registry commands
2. NTLM Reflection — Complete AD kill chain from low-priv → domain compromise
3. Ligolo-MP Pivoting — Modern pivoting with double-hop and loopback routing
4. DLL Execution via LOLBins — 7 LOLBin loaders with exact command syntax
5. Responder Guide — LLMNR/NBT-NS poisoning with MultiRelay shell access
6. Credential Dumping Cheatsheet — LSASS, SAM, DCSync, WDigest
7. Pass-the-Hash Lateral Movement — Multiple PTH variants
8. Reverse Shell Generation — Multi-platform payload generation

**85+ Techniques Extracted** with **42 MITRE ATT&CK Mappings**

**Estimated Training Pairs:**
- From 8 scraped articles: ~280 pairs
- Full category (projected): 1,250–1,700 pairs

**Note:** This is the largest untapped source. Only 8 of ~414 articles were scraped. A full scrape would yield significantly more data.

---

## Part 2: AI Red Team Tools (25 Tools Analyzed)

### HIGH PRIORITY — Immediate Data Goldmines

| Rank | Tool | Stars | Est. Entries | License | Key Data |
|------|------|-------|--------------|---------|----------|
| **1** | **Promptfoo** | 22k | 10,000+ | MIT | 40+ redteam plugins, 15+ strategies, prompt injection, jailbreaks, harmful content, PII, politics, cybersecurity |
| **2** | **Garak** (NVIDIA) | 8.1k | 5,000–10,000 | Apache-2.0 | 40+ probe modules, resources/ directory, DAN 1.0–11.0, promptinject, smuggling |
| **3** | **The Big Prompt Library** | 5.1k | 550+ | MIT | System prompts (25+ providers), Jailbreak/ folder, Security protections, CustomInstructions |
| **4** | **Promptmap** | 1.2k | 150+ | GPL-3.0 | Clean YAML rules: distraction, prompt_stealing, jailbreak, harmful, hate, social_bias |
| **5** | **PyRIT** (Microsoft) | 3.9k | 100+ | MIT | Prompt converters (atbash, caesar, morse, codechameleon, noise, translation), jailbreak templates |
| **6** | **FuzzyAI** (CyberArk) | 1.5k | 620+ | Apache-2.0 | harmful_behaviors.csv (520), adv_prompts.txt, adv_suffixes.txt, persuasion_taxonomy.jsonl, pandoras_prompts.txt |

### MEDIUM PRIORITY

| Tool | Stars | Est. Entries | Key Data |
|------|-------|--------------|----------|
| AgentDojo (ETH Zurich) | 605 | 100+ | Agent prompt injection benchmark tasks |
| Giskard | 5.4k | Unknown | Built-in injection payload dataset (v3 scan) |
| Moonshot (AI Verify) | 331 | Unknown | Benchmark datasets + attack recipes |
| HouYi (NTU) | 266 | ~20 | Automated injection framework examples |
| Parseltongue | 571 | ~50 patterns | Text transformation/obfuscation patterns |

### LOW / NOT SUITABLE

| Tool | Reason |
|------|--------|
| ART, CleverHans, AIX360, AIF360, Captum, LIT | ML security for images/tables/audio — no LLM prompt data |
| NeMo Guardrails | Defensive toolkit (guardrails), not attack data |
| Inspect Framework | Evaluation framework, not attack prompts |
| LLMBUS, TokenBuster, aiapwn, AnyCoder | Small tools, minimal data |
| ZetaLib | Has interesting folder names but minimal README — worth a closer look |

### Estimated Training Pairs from AI Tools

| Category | Tools | Est. Pairs |
|----------|-------|------------|
| Prompt Injection (GOLD) | Promptfoo, Garak, Promptmap, PyRIT, AgentDojo, HouYi, FuzzyAI, Giskard | ~12,000 |
| Jailbreak/Attack Prompts | Promptfoo, Garak, Big Prompt Library, FuzzyAI, PyRIT, ZetaLib(?), Parseltongue | ~15,000 |
| Benign/Contrast Prompts | FuzzyAI (benign_prompts.txt) + tool-generated | ~5,000 |
| System Prompts | Big Prompt Library | ~500 |
| Prompt Obfuscation | Promptfoo strategies, Parseltongue, PyRIT converters | ~500 patterns |
| **TOTAL** | | **~33,000+** |

---

## Part 3: Integration Plan for AttackLM Data Pipeline

### Current Pipeline (from GAMEPLAN.md)

```
Step 1: clone_repos.sh → atomic-red-team, stockpile, sigma
Step 2: extract_by_tactic.py → data/manifests/{tactic}.json
Step 3: generate_dataset.py → data/datasets/{tactic}_dataset.jsonl
Step 4: generate_orchestrator.py → orchestrator_dataset.jsonl
Step 5: train_template.py → QLoRA fine-tuning
```

### New Data Sources to Integrate

#### Phase 1: Red Team Tactics (Websites) → Tactical Agent Training

These map directly to the existing MITRE ATT&CK tactic structure:

| Source | Maps To | New Pairs |
|--------|---------|-----------|
| InternalAllTheThings AD section | CredentialAccessAgent (TA0006), LateralMovementAgent (TA0008) | 1,500–2,000 |
| InternalAllTheThings PrivEsc section | PrivilegeEscalationAgent (TA0004) | 800–1,200 |
| InternalAllTheThings Persistence section | PersistenceAgent (TA0003) | 500–700 |
| InternalAllTheThings Defense Evasion | DefenseEvasionAgent (TA0005) | 300–500 |
| rgbwiki Red Cell (all sections) | All tactical agents | 800–1,200 |
| HackingArticles (full scrape) | All tactical agents | 1,250–1,700 |

**New script needed:** `extract_web_to_jsonl.py` — scrapes websites and converts to JSONL format matching existing dataset structure.

#### Phase 2: AI Red Team Tools → New "PromptInjectionAgent"

These tools provide data for a **new capability** not in the current architecture:

| Source | Data Type | New Pairs |
|--------|-----------|-----------|
| Promptfoo, Garak, Promptmap, PyRIT, FuzzyAI | Prompt injection attacks | ~12,000 |
| Big Prompt Library, FuzzyAI, Garak | Jailbreak/attack prompts | ~15,000 |
| FuzzyAI | Benign prompts (contrastive) | ~5,000 |
| Big Prompt Library | System prompts | ~500 |

**New agent proposed:** `PromptInjectionAgent` — specializes in LLM prompt injection, jailbreak crafting, and AI system exploitation.

**New script needed:** `extract_ai_tools_to_jsonl.py` — clones AI tool repos, extracts prompt datasets, converts to JSONL.

#### Phase 3: Orchestrator Enhancement

The orchestrator dataset can be enriched with:
- Multi-step attack chains from InternalAllTheThings (e.g., "enumerate AD → kerberoast → crack ticket → pass-the-hash → DCSync")
- AI red team scenarios (e.g., "identify LLM endpoint → test guardrails → craft jailbreak → extract system prompt")

---

## Part 4: Recommended Action Plan

### Immediate (This Session)

1. ✅ **Research complete** — all four sources analyzed
2. 🔲 **Clone HIGH-priority AI tool repos** to `data/ai_tools/`:
   - `git clone` Promptfoo, Garak, Big Prompt Library, Promptmap, PyRIT, FuzzyAI
3. 🔲 **Write `extract_web_to_jsonl.py`** — scraper for InternalAllTheThings + rgbwiki + HackingArticles
4. 🔲 **Write `extract_ai_tools_to_jsonl.py`** — extractor for AI tool prompt datasets
5. 🔲 **Full scrape of HackingArticles** — 414 articles (only 8 done so far)

### Short-Term (Next Session)

6. 🔲 Run extraction scripts → populate `data/datasets/` with new JSONL files
7. 🔲 Generate orchestrator routing data for prompt injection scenarios
8. 🔲 Train initial models with expanded datasets
9. 🔲 Evaluate quality of web-scraped vs tool-extracted training data

### Estimated Total Training Data After Expansion

| Source | Before | After |
|--------|--------|-------|
| Atomic Red Team + Stockpile + Sigma | ~5,000–10,000 | ~5,000–10,000 |
| InternalAllTheThings | 0 | +3,500–5,000 |
| rgbwiki Red Cell | 0 | +800–1,200 |
| HackingArticles (full) | 0 | +1,250–1,700 |
| AI Red Team Tools | 0 | +33,000+ |
| **TOTAL** | **~5,000–10,000** | **~43,550–50,900** |

---

## Part 5: Key Files to Create

### New Scripts Needed

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/extract_web_to_jsonl.py` | Scrape red team websites → JSONL | URLs + CSS selectors | `data/datasets/web_{source}_dataset.jsonl` |
| `scripts/extract_ai_tools_to_jsonl.py` | Extract AI tool prompt datasets → JSONL | Cloned repos in `data/ai_tools/` | `data/datasets/prompt_injection_dataset.jsonl` |
| `scripts/full_scrape_hackingarticles.py` | Paginate through all 46 category pages, fetch all ~414 articles | hackingarticles.in/penetration-testing/ | `data/datasets/hackingarticles_dataset.jsonl` |

### New Data Directories

```
data/
├── ai_tools/           # Cloned AI red team tool repos
│   ├── promptfoo/
│   ├── garak/
│   ├── TheBigPromptLibrary/
│   ├── promptmap/
│   ├── PyRIT/
│   └── FuzzyAI/
├── web_scrapes/        # Raw scraped HTML/JSON from websites
│   ├── internalallthethings/
│   ├── rgbwiki/
│   └── hackingarticles/
└── datasets/           # Existing + new JSONL datasets
    ├── prompt_injection_dataset.jsonl    # NEW
    ├── jailbreak_dataset.jsonl           # NEW
    ├── web_tactics_dataset.jsonl         # NEW
    └── ... (existing tactic datasets)
```

---

## Appendix: MITRE ATT&CK Coverage Map

| Tactic ID | Agent | Existing Coverage | New Coverage from Web Sources |
|-----------|-------|-------------------|-------------------------------|
| TA0002 | ExecutionAgent | Atomic + Stockpile | InternalAllTheThings (50+ techniques), HackingArticles (LOLBins, reverse shells) |
| TA0003 | PersistenceAgent | Atomic + Stockpile | InternalAllTheThings (30+ mechanisms), rgbwiki (3+) |
| TA0004 | PrivilegeEscalationAgent | Atomic + Stockpile | InternalAllTheThings (50+), rgbwiki (13+), HackingArticles (UAC bypass) |
| TA0005 | DefenseEvasionAgent | Atomic + Stockpile | InternalAllTheThings (AMSI, obfuscation), rgbwiki (AMSI bypass) |
| TA0006 | CredentialAccessAgent | Atomic + Stockpile | InternalAllTheThings (DCSync, Kerberoasting, SAM), rgbwiki (Mimikatz, LSASS) |
| TA0007 | DiscoveryAgent | Atomic + Stockpile | InternalAllTheThings (BloodHound, enumeration), rgbwiki (enumeration) |
| TA0008 | LateralMovementAgent | Atomic + Stockpile | InternalAllTheThings (PTH, PTT, RBCD), rgbwiki (PsExec, RDP, SSH) |
| TA0011 | CommandAndControlAgent | Atomic + Stockpile | InternalAllTheThings (C2 frameworks), rgbwiki (C2 cheatsheets) |
| **NEW** | **PromptInjectionAgent** | None | AI tools: 33,000+ prompt injection + jailbreak pairs |
