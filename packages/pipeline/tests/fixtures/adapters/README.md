# Adapter fixtures (WS-C)

Captured 2026-11-16 from the hosts named in
`docs/product-development-cycle/06-india-adapter-spec.md`. Tests never touch the
network; assertions run against these bytes.

| File | Source id | Capture | sha256 |
|---|---|---|---|
| `ugc_university_central.json` | `ugc_university_list` | Live `GET Getuniversity_details?unitypeID=1` (200, 52,379 B, 57 rows). **PII values redacted** to `REDACTED`; the leaked keys are retained so the drop-list test is honest. | `ce342d9cf8def805406ba4251ff552ce77f93078c8ea8c58faac97d5a979c425` |
| `ugc_error_shell.html` | `ugc_consolidated_lists` / UGC trap | Live `GET /oldpdf/` (HTTP 200, 51,908 B, `text/html`). Not a document. | `f29a4f8179d3f37c8a721747e0e743d136962870d6dc5e15bbafb45a5e446f18` |
| `naac_accreditationresults_snippet.html` | `naac_results` | Snippet of live page (full: 200, 556,679 B, 1,539 PDF anchors). The linked `SC_258_Health_Sciences.pdf` is born-digital and carries AISHE code `U-0523`, CGPA `3.67`, grade `A++` and the 258th Standing Committee date. | `9da658236bc6cc89bf364eb7f46c5ef3579467e0e35e92b67a98caa834a8577c` |
| `nabh_get_hospitals_page1.json` | `nabh_hospitals` | **Trimmed + redacted** live `POST admin-ajax.php action=get_hospitals&page=1` (200, 107,350 B): first 3 of 25 rows, real `pagination` (866 pages / 21,637 results), 3 `mapData` markers. Contact column (phones/websites) and presigned URLs removed; certificate URLs query-stripped. | `26e8f5d8567e69cedfa85e9194b1efb0cfb9ec1cda3b940a5d2db0c1d9366a5c` |
| `aishe_final_report_index_snippet.html` | `aishe_reports` | Snippet of the index; static xlsx path recorded but **not fetch-verified** (connection 000). | `f34f50b34f51f78055d723185c099b386be4e5dc429544845541b2fc80fa0fe2` |
| `state_counselling_notices_snippet.html` | `counselling_maharashtra` | Snippet of the live WordPress notice listing. | `ac9baee23c9dcdf76a69e88f0d9c7ea7a1a49bea9fb6e425fc30c2fcaa882baa` |
| `institution_site_nirf_listing_snippet.html` | `institution_sites` | Live `https://mtct.ac.in/nirf/` (200, 99,312 B): anchor list trimmed to the NIRF/placement documents plus the IQAC/NAAC navigation and a login link. | `e7b5bd1778eead3badb007cd806df45c5eaecb4b0ebec7fc7f9c85f368f09414` |
| `health_university_ruhs_listing_snippet.html` | `state_health_universities` | Live `https://ruhsraj.org/affiliated-colleges/` (200, 439,746 B): first 40 of 573 `<option>` college names plus 5 PDF anchors. The real college list is a `<select>`, not a table. | `5d8a3d3363d9abdc8e4e13188108a5dd0301fc77b72c046c81bb842a8443011e` |
| `health_university_tnmgrmu_sitemap_snippet.xml` | `state_health_universities` | Live `https://www.tnmgrmu.ac.in/wp-sitemap-posts-page-1.xml` (200, 99,887 B): 43 of 328 URLs, including `/affiliated-colleges-institutions/`, `/affiliation/seat-matrix/`, `/academics/courses-offered/allied-health-sciences/`. | `dee41e943cbde5e51f68649107bb19834f95a3ec4b2d6e20075886aba1d4e171` |
| `fee_regulator_mhafra_form.html` | `fee_regulators` | Live `https://ay25-26.mahafraportal.org/ssi_prp_24/outer.php?q=fee_search_report` (200, 13,875 B): the search form only. `sub_type` is JS-populated, so this is the form-replay abstention fixture. | `3da853f84d3186534a0459676fca6d5411ee417a02afdb0246856a8d88f8b427` |
| `fee_regulator_order_listing_structural.html` | `fee_regulators` | **Structural sample, not fetched**: no state published a machine-readable fee-order index that could be verified on 2026-11-16 (spec §4.6.1). Reproduces the observed listing shape (dated order PDFs) to test the generic extractor; contains no real order or amount. | `a9f38b369e1224ad9642ad474c00df569ed6d0aa02e35490f3f9e456de8c3011` |
| `state_council_ncahp_statecouncil_snippet.html` | `state_paramedical_councils` | Live `https://ncahp.abdm.gov.in/StateCouncil` (200, 36,904 B): all 72 `statecouncil/*.pdf` notification and rules anchors with their date labels. | `4ccb825b9103a3c8bc512a1760fde35dbb44a8672a5c60835cd1049187dba68e` |
| `ncahp_regulator_listing.html` | `ncahp` | **Structural sample, not fetched**: models the AHP-registration recognising list at https://ncahp.abdm.gov.in/ (adapter `ncahp`): PDF recognition notices, a duplicate link, and non-PDF/JS anchors. No institution is real. | N/A |
| `nirf_2025_medical_snippet.html` | `nirf_rankings` | Live `GET /Rankings/2025/MedicalRanking.html` (200, 183,858 B, **browser UA required**; default curl UA gets CloudFront 403): header plus the first 3 of 151 table rows. | `5d62065bf1c86d89ef9a73365cca96ac447ae3faf8d854146177688299a91864` |
| `js_shell_sample.html` | `state_health_universities` (shell detector) | Live `https://mahafra.org/` (200, 875 B): a React shell with no server-rendered anchors. Used to test the generic JS-shell abstention, not as institution content. | `d29111c07416ea81997f6b782e2e55bff9011612b795afe50d22db2a6a114e3b` |

Raw-byte hashes for the unmodified captures match the values above; the UGC
JSON hash is for the redacted fixture (raw capture, before redaction:
`48f39d2c9592558c7e4abb191286d8e4134941a8d2defeeff0498ee8fd38ef6c`).
