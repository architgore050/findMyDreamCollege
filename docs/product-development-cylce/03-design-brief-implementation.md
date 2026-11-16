**Document 3: Design Brief for Implementation (Phases 3–4)**  

These are based on extensive research into BSc/Bachelor of Medical Laboratory Technology (MLT) / Medical Laboratory Science (MLS) programs (the actual target field; “MLT nursing” maps to this allied-health paramedical bachelor’s, typically 3–4 years including mandatory clinical internship/placement, distinct from pure nursing but frequently offered by the same institutions).  

Key research findings: Graduates work as Medical Laboratory Technologists/Scientists, performing diagnostic tests (hematology, microbiology, biochemistry, blood bank, etc.). In India, entry salaries are typically ₹2.5–4 LPA (fresher), rising to ₹6–10+ LPA with experience; abroad pathways (Gulf via Prometric/DataFlow, Australia via AIMS, Canada via CSMLS, US via ASCP, UK via HCPC/IBMS) offer substantially higher pay but require credential evaluation + licensing exams. “Placement” for this field is primarily the mandatory clinical internship + subsequent employment rate and salary outcomes (bottom quartile relative to local cost of living is the right metric, as official placement reports are often incomplete or gamed). Data sources include government university lists, accreditation bodies (NAACLS, AIMS, CSMLS, HCPC, etc.), placement reports, Numbeo CoL/QoL indices, visa/post-study work rules, and study-abroad platforms.  

---

### Document 3: Design Brief for Implementation Agents (Phases 3 & 4)

**Product Vision**  
A generalized, data-driven decision tool that scores universities/programs offering bachelor’s-level MLT/MLS (or closely equivalent Biomedical Laboratory Science) on a multi-factor model covering (1) quality & outcomes, (2) accessibility for the applicant, and (3) full budget realism. Output prioritizes the highest-signal insights and presents a short, explainable shortlist rather than exhaustive lists. All intermediate ML/NLP artifacts must be reviewable.

**Geographic Scope – Top ~50 Countries (must include India)**  
Prioritize countries with (a) actual bachelor’s MLT/MLS programs, (b) usable public data or ranking presence, (c) reasonable international-student pathways or strong domestic options, (d) post-study work or employment demand for lab professionals, and (e) CoL/QoL data availability. Suggested initial list (refine with program-count scraping):  

India (mandatory baseline), Australia, Canada, United States, United Kingdom, New Zealand, Ireland, Germany, Netherlands, Singapore, Malaysia, UAE, Saudi Arabia, Qatar, Oman, Kuwait, South Africa, Philippines, Japan, South Korea, Taiwan, Hong Kong, China (selected high-quality programs), Thailand, Indonesia, Vietnam, Nepal, Pakistan, Bangladesh, Sri Lanka, Egypt, Ghana, Nigeria, Kenya, France, Spain, Italy, Sweden, Finland, Denmark, Norway, Switzerland, Austria, Belgium, Portugal, Poland, Czechia, Hungary, Turkey, Israel, Brazil, Mexico.  

Rationale: These cover high-outcome destinations (Australia/Canada/NZ/UK/US/Gulf), strong domestic Indian options, affordable Asia/Africa options with programs, and European alternatives with good QoL or tuition advantages. Exclude pure low-signal countries lacking programs or data. Update the list periodically via government higher-education portals and Educatly/Bachelorsportal-style aggregators.

**Core Data Pipelines (generalized, not hardcoded)**  
1. **University & Program Inventory**: Scrape government higher-ed lists, accreditation bodies, and public aggregators for bachelor’s MLT/MLS/Biomedical Lab Science programs. Capture alternate names/acronyms.  
2. **Quality & Reputation NLP**: Public mentions, rankings (QS Life Sciences & Medicine, subject proxies), student/alumni reviews → topic scores (teaching, facilities, research, support).  
3. **Outcomes Mining (LLM-assisted)**: Placement/internship reports (government-mandated where available). Extract employment rate, salary distributions. Prefer bottom 25th percentile salary; fall back to estimates with confidence flags. Adjust by regional CoL (Numbeo or equivalent). Flag gaming (high non-participation).  
4. **Living Quality**: Country + city/region QoL, safety, healthcare access, pollution, climate (Numbeo + official indices). Separate student-specific signals.  
5. **Admissions Feasibility**: Public exam requirements, eligibility (marks, age, subjects), registration status (live scrape of deadlines), location of exam (filter pure in-country only unless remote options exist), qualitative factors (prompt user 1–5 on extracurriculars, leadership, etc.). Link to past-paper repositories where public. International-student acceptance signals and English requirements.  
6. **Budget Engine**: Tuition + estimated equipment/materials + living (hostel/mess vs independent PG + food + utilities + transport) + visa/insurance + misc. Toggleable hostel vs independent. Currency conversion + confidence bands.  
7. **Country Welcome & Mobility**: Post-study work length, shortage-list status for MLT, PR pathways, historical Indian student visa success (public aggregates).

**Scoring Model (high-level)**  
Composite score with transparent sub-scores and weights (user-adjustable later):  
- Outcomes (employment + CoL-adjusted bottom-quartile salary + licensure pathway strength) – highest weight.  
- Program quality & clinical training.  
- Accessibility (exam feasibility + international friendliness + user profile match).  
- Net budget realism (total cost vs expected early earnings).  
- QoL at study + placement locations.  

Present top insights first (the few factors that explain most of the variance for that shortlist). Save all intermediate scraped/NLP/LLM outputs for review.

**User Inputs & Outputs**  
- Inputs: Academic marks, self-rated qualitative factors (1–5), budget ceiling, preferred regions, risk tolerance (e.g., accept estimated data).  
- Outputs: Ranked shortlist with explainable scores, budget breakdown (toggleable), next-action checklist (exams still open, past papers, application links), confidence flags, and “why this one” narrative.  

**Technical & Process Constraints for Implementation**  
- Prefer public government and regulatory sources.  
- Partial answers + confidence > perfect but sparse accuracy.  
- Dynamic (not hardcoded) deadline and registration status.  
- Modular so new countries or data sources can be added.  
- All ML artifacts reviewable.  
- Privacy: minimal personal data stored.  
- Phase 4 testing: validate against known good/bad Indian MLT programs, known high-outcome Australian/Canadian programs, and real placement report samples; user testing with the target family profile.

**Success Criteria for the Built Product**  
- Covers the prioritized ~50 countries with usable scores.  
- Surfaces MLT-specific outcome realities (including bottom-quartile and CoL adjustment).  
- Correctly filters inaccessible exams.  
- Produces actionable, reviewable recommendations that a non-expert family can trust and act on.  
- Transparent about data limitations.

This brief gives implementation agents a clear, research-grounded scope that avoids low-value global scraping while addressing the full original requirements plus the critical missing factors identified in Phase 1.
