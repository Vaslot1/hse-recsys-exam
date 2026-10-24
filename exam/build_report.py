"""Build the evidence-based exam report; uses saved results only."""
from pathlib import Path
import json, shutil, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'exam'; OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
D=E/'derived/report'; D.mkdir(exist_ok=True)
shutil.copy2('C:/Users/vaslot/AppData/Local/Temp/codex-clipboard-cfc49683-c97e-4a1d-9a11-ef59b17ba66f.png',D/'kaggle-screenshot.png')
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('Arial-Bold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
s=getSampleStyleSheet()
for name in ['Normal','BodyText','Title','Heading1','Heading2']:s[name].fontName='Arial'
s['BodyText'].fontSize=9;s['BodyText'].leading=12;s['BodyText'].spaceAfter=7
s['Title'].fontSize=19;s['Title'].leading=23;s['Title'].textColor=colors.HexColor('#143d59')
s['Heading1'].fontSize=13;s['Heading1'].leading=16;s['Heading1'].spaceAfter=9
s['Heading2'].fontSize=10;s['Heading2'].leading=13
s.add(ParagraphStyle(name='Cell',fontName='Arial',fontSize=7.5,leading=9.5))
s.add(ParagraphStyle(name='Small',fontName='Arial',fontSize=8,leading=10,spaceAfter=6))
story=[]; textlog=[]
def p(t,style='BodyText'):story.append(Paragraph(t,s[style]));textlog.append(t)
def h(t):p(t,'Heading1')
def table(rows,widths):
 t=Table([[Paragraph(html.escape(str(c)),s['Cell']) for c in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8098aa')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e1e6'))]));story.append(t);textlog.append(str(rows))
def page():story.append(PageBreak())
W=499
p('From broad viewing labels to preference-focused TV recommendations','Title')
p('Valerii Bobrov | Individual | vmbobrov@edu.hse.ru<br/>GSB final exam: SWEET.TV recommendation | 24 October 2026','Small')
p('Repository: <link href="https://github.com/Vaslot1/hse-recsys-exam">https://github.com/Vaslot1/hse-recsys-exam</link>','Small')
h('Abstract')
p('Predicting five preferred television programmes requires distinguishing incidental viewing from repeated high-coverage watching. This study compares training-only popularity, implicit-feedback ALS and Personalized PageRank on a fixed temporal split, then diagnoses relevance-label mismatch and changes only the PPR restart profile. The selected coverage-count restart with damping 0.25 achieves validation MAP@5 of 0.367005 across 1,260 users, while the supplied Kaggle screenshot shows a highest private score of 0.34776. Preference-focused labels and restart weights support this offline strategy, but user-level failures, tie ambiguity and repeated validation selection limit generalization claims.')
h('1. Problem, sources and alternatives')
p('The task is a ranked Top-5 for weeks 31-42, using channel viewing intervals and programme schedules [1]. A five-minute fragment of a one-hour broadcast can pass the initial interaction rule but fails the competition-described 80% coverage rule. The contributions are an auditable interval join, a common comparison of two eligible methods plus popularity, and a controlled correction of the restart profile. Implicit ALS separates binary preference from confidence [2]; personalized PageRank permits a nonuniform restart distribution [3]. Content-based filtering was considered for cold items but not implemented; rank fusion was deferred to avoid adding untested components.')
h('2. Data and leakage controls')
p('The log contains 5,821,987 rows. Matching preserves exact IDs, joins equal channel IDs, unions duplicate/overlapping views, accumulates disjoint visits within each airing, and removes ambiguous schedule seconds. ID 0 is unlabelled, never a candidate or an explicit dislike. The original threshold is O ≥ min(300, 0.5L), with full airing duration L. There are 17 duplicate view rows and 168 excluded unique views: 117 duration mismatches and 52 over-24-hour sessions, with one overlap. Saved qualifying matches total 1,233,106 for training and 859,214 for validation.')
p('Fit: 9 March-18 May 2020; validate: 18 May-27 July; predict: 27 July-19 October (all end-exclusive, source wall-clock time). Every graph, factor and popularity count uses training only. Future schedules provide availability, not future viewing. Validation has 6,384 candidates; the future pool has 6,632. Repeats are allowed because programme IDs can recur across airings; each list has five distinct nonzero IDs. The same training distinct-viewer popularity supplies fallback and list completion.')
h('3. Methods and ranking')
p('Popularity ranks distinct training viewers. ALS minimizes confidence-weighted squared preference error plus L2 regularization; the selected run uses binary confidence, alpha 1, regularization 10, 32 factors, 15 iterations and seed 42. Its score is a user-item dot product. PPR uses the unchanged 2,901-node programme graph with distinct-user co-view edge weights and no self-loops. The final restart is each user’s normalized count of training airings with O/L ≥ 0.8; damping is 0.25. PPR mass ranks graph-supported candidates, with popularity/numeric-ID tie breaks and fallback. These scores are ranking quantities, not interchangeable probabilities of watching.')
page();h('4. Evaluation, diagnosis and model selection')
p('For a user with target set R, AP@5 = Σ(k=1..5) [hit(k) × hits-through-k/k] / min(5, |R|); empty targets score zero. MAP is the equal-user mean. This is binary membership, not positional agreement with the target order [3]. The original broad labels counted every qualifying programme. The aligned diagnostic retains airings with at least 80% coverage and selects the five most frequent programme IDs, breaking cutoff ties by ascending numeric ID. Fragmented visits count once per airing; different airings count separately. These conventions are explicit local assumptions, not a reconstruction of undocumented hidden code.')
table([['Method','Broad-label MAP','Top-five MAP','≥5-qualified cohort'],['Popularity','0.729399','0.258167','0.269742'],['Tuned ALS','0.632365','0.182209','0.190922'],['Uniform PPR 0.85','0.737352','0.263313','0.275014'],['Uniform PPR 0.50','0.738275','0.264163','0.276162'],['Count restart 0.50','Not evaluated','0.347120','0.367606'],['Selected coverage restart 0.25','Not evaluated','0.367005','0.388145']],[175,108,108,108])
p('Full-cohort results use all 1,260 users, including 22 empty strict targets. The fixed auxiliary cohort has 1,150 users with at least five 80%-qualified programmes; it excludes 110 users and is not the primary selection criterion. Broad labels had 162,418 relevant user-item pairs; strict top-five labels have 6,005. Frozen predictions reproduce the old scores before diagnostic rescoring. The ALS configuration was tuned under the original labels; it was not retuned for the corrected target, limiting claims about the best possible ALS model.')
table([['Final experiment','Full-cohort MAP@5'],['All-count restart: damping 0.05','0.347174'],['All-count restart: damping 0.10','0.347158'],['All-count restart: damping 0.25','0.347179'],['All-count restart: damping 0.50 (reused)','0.347120'],['≥80%-coverage counts: damping 0.25','0.367005']],[355,144])
p('Damping 0.25 won the all-count comparison by a very small margin. Holding that damping and the graph fixed, coverage-filtered restart improved MAP by 0.019826. All five configurations completed in 210.344 seconds inside the 480-second budget. The final model converged for 1,242 users (at most eight iterations); 18 used the unchanged popularity fallback because no strict restart history remained.')
p('Twenty saved shared seeded cutoff-tie resolutions (seeds 0-19) give selected-model mean MAP 0.348041, observed range 0.344756-0.350905. Paired improvement over count-restart 0.50 is positive in 20/20 resolutions, mean +0.017478. This is empirical sensitivity, not exact bounds, a confidence interval or an independent test. The winner was selected by fixed-label full-cohort MAP, not by choosing a favorable tie seed.')
p('Verification: 86 small tests passed in the final audit, covering known-answer matching/MAP, ALS numerical solves, transition row sums, dangling repair, mass conservation, convergence and CSV checks. The selected run’s maximum residual is 3.406×10^-12 and mass error 2.953×10^-14. Saved artifact hashes and both cohort metrics were rechecked. Recorded software: Python 3.12.10, NumPy 2.4.2, SciPy 1.17.1; execution used Windows. CPU/RAM and exact historical revisions are not fully recorded; runtimes are not controlled hardware benchmarks. These are agent checks, not a claim of personal student verification.','Small')
users=json.loads((D/'two-users.json').read_text(encoding='utf-8'))
for n,u in enumerate(users):
 page();uid=u['history']['user_id'];h(f'5.{n+1}. Same-user recommendation comparison')
 p(f'<b>User {uid}</b>. These two cases were originally selected after examining the uniform-PPR results: a strong gain and a strong loss relative to popularity. Keeping them fixed prevents replacing inconvenient examples after the final change; they are diagnostic cases, not a representative sample.')
 hist=u['history'];p(f"Original training history: {hist['training_broadcasts']} qualifying broadcasts across {hist['training_distinct_programmes']} programmes. Category counts: "+'; '.join(f'{k}: {v}' for k,v in hist['training_category_broadcasts'].items())+'.')
 p('Frozen targets: '+', '.join(u['targets'])+'. Hit below means membership in this strict target set. Train is the original qualifying training-broadcast count. Titles/categories are representative historical schedule labels, not episode predictions. The new PPR scores were recalculated only for these two profiles from the saved graph; no model was fitted.')
 rr=[['Method / rank','Programme ID / title','Category','Score','Train','Hit']]
 for r in u['recommendations']:
  method={'popularity':'Popularity','tuned_als':'ALS','selected_ppr':'PPR 0.25'}[r['method']]
  score=float(r['score']);rr.append([f"{method} / {r['rank']}",r['programme_id']+' / '+r['title'],r['category'],f'{score:.5g}',r['training_broadcasts'],'Yes' if r['validation_hit'] else 'No'])
 table(rr,[70,211,70,55,40,53])
 p('AP@5: '+ '; '.join(f'{m} = {v:.2f}' for m,v in u['ap'].items())+'.')
 if n==0:
  p('This history is dominated by children’s programmes (21 of 28 broadcasts). Generic popularity misses all five strict targets. ALS scores 0.52 and selected PPR 0.42: personalization helps this viewer, but the aggregate winner is not the best individual method. The restart mechanism emphasizes repeated high-coverage training interests; it cannot guarantee that their future frequency ordering persists. The table supports familiar-preference matching, not a claim that the viewer enjoyed a programme.')
 else:
  p('This history is broad and film-heavy (68 of 83 broadcasts), with 77 distinct programme IDs. Only two programmes survive as strict validation targets, so this user belongs to the full cohort but not the ≥5-qualified cohort. All three lists miss both targets. Recommending familiar or co-view-related programmes does not solve sparse, changing future preference evidence; a miss is not proof of dislike. The failure limits any claim that the final system works uniformly across viewers.')
page();h('6. Kaggle evidence and platform decision')
p('The supplied submission-page screenshot shows three completed late submissions. The highest displayed private score is <b>0.34776</b>, paired with public <b>0.35400</b>; earlier private scores are 0.32706 and 0.28564. This report uses the highest displayed result as its designated score evidence. The screenshot does not expose a submission ID or CSV hash, so exact byte-level attribution to the selected local file cannot be independently proved from this image alone. Its visible selection boxes are unchecked; no manual Kaggle final-selection action is claimed.')
story.append(Image(str(D/'kaggle-screenshot.png'),width=W,height=W*1240/2105));textlog.append('Figure: original user-supplied Kaggle screenshot, preserved without alteration.')
p('Figure 1. User-supplied Kaggle submission page: Russian full name and Vaslot1 are visible. The image is preserved without editing. The latest local candidate has 1,260 users and five distinct future-eligible IDs each. SHA-256: <font size="7">16dae00a24730568033467a8e09aec4dc4b63e963888b064c74277b4d48792de</font>.','Small')
p('<b>Decision.</b> Select coverage-count restart PPR 0.25 with the existing popularity fallback for an offline deployment candidate. It leads the fixed validation comparison, reuses the graph and finishes within the budget. Its quality gain is not a demonstrated retention or revenue gain. The original score gap was largely label-definition sensitivity: uniform PPR falls from 0.738275 to 0.264163 when only labels change. This explains why broad-label validation was misleading, but does not prove that labels account for every Kaggle difference.')
p('<b>Limitations and next step.</b> Only 10.8% of first-schedule rows have nonzero IDs; this is a row fraction, not measured airtime coverage. Timestamp timezone, hidden fragmentation and tie rules remain uncertain. Cold items lack learned graph/factor support, quarantine changes usable evidence, and validation has been reused for tuning. The same two users reveal both personalization benefit and a complete failure. A next experiment would evaluate the fixed selected configuration on a new earlier rolling temporal holdout before further tuning; it has not been run.')
p('<b>AI usage and trust boundary.</b> Codex assisted with code, tests, debugging, source lookup, analysis and this report/PDF. The exact historical model versions are not reliably recorded. The student directed protocol changes and ran the full experiments; automated tests, file checks and two-user arithmetic reported here were performed by the assistant. No personal hand calculation, independent paper reading or independent numerical verification is asserted. Personal review remains necessary. The original native AI-session export must accompany the report; this PDF and results JSON are not substitutes. No live A/B test or revenue result is claimed.','Small')
page();h('References and evidence index')
p('Sources checked during report preparation, accessed 24 October 2026. External sources support algorithm definitions, not the experiment’s measured results. The GitHub URL was supplied by the student; remote contents were not independently verified.','Small')
refs=[
'[1] S. Jin, “Final Exam Quiz (GSB): TV Program Recommendation on sweet.tv,” course brief and LLM4Rec Midterm Quiz slides, HSE, 2026. Local sources: exam/exam_quiz_GSB.md and exam/LLM4Rec. Midterm Quiz.pptx. Dataset-target description supplied in the session; hidden implementation unavailable.',
'[2] Y. Hu, Y. Koren, and C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” IEEE ICDM, 2008. Author-hosted text checked via indexed source: <link href="https://yifanhu.net/PUB/cf.pdf">https://yifanhu.net/PUB/cf.pdf</link>. Used for confidence-weighted implicit-feedback modelling, not as evidence of our measured performance.',
'[3] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to Information Retrieval, Cambridge University Press, 2008, Sections 8.4 and 21.2.3. Checked online sections: <link href="https://nlp.stanford.edu/IR-book/pdf/08eval.pdf">Evaluation in information retrieval</link> and <link href="https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html">Topic-specific PageRank</link>. The report’s cutoff-five normalization is the explicitly implemented local protocol.',
'[4] S. Jin, “LLM4Rec Homework Guidelines,” and “HW6 Class Feedback - GSB Cohort,” course materials, 2026. Local files: homework_guidelines.pdf and hw-feedback/HW6_class_feedback.md. Used for verification, disclosure and reporting requirements.'
]
for r in refs:p(r)
h('Reproducible evidence paths')
for t in [
'Data quality: exam/derived/matching/data-quality.json; original-source hashes: tmp/sweet-tv-audit/verification.json.',
'Method implementations: exam/matching.py, evaluation.py, als.py, ppr.py. Original configurations and commands: exam/MATCHING.md, ALS.md, EVALUATION.md, PPR.md.',
'Frozen comparison: exam/derived/label-diagnostic/comparison.csv. Original ALS tuning: exam/derived/als-tuning/comparison.csv. Keep the broad-label and strict-label results separate.',
'Final experiment: exam/derived/ppr-final-8min/results.csv, context.json, run-status.json and selection.json. Selected checkpoint: completed/coverage80-d025/{result.json, diagnostics.json, validation-predictions.csv, tie-summary.json, submission.csv}.',
'Two-user report evidence: exam/derived/report/two-users.json. Previous comparison and histories: exam/derived/business-analysis/. Scores retain their native method scale.',
'Verification command: python -B -m unittest discover -s exam/tests -q. Final experiment command (already completed): python -B exam/final_ppr_experiment.py --budget-seconds 480 --output-dir exam/derived/ppr-final-8min. Existing experiment output directories are preserved.',
'Outstanding delivery evidence: native unmodified AI-session export; remote repository contents; exact correspondence between scored upload bytes and the local CSV. These are not replaced by a claimed test pass.'
]:p(t,'Small')
def footer(c,d):
 c.setFont('Arial',8);c.setFillColor(colors.HexColor('#536674'));c.drawString(48,28,'SWEET.TV | GSB final exam | Valerii Bobrov');c.drawRightString(547,28,str(d.page))
path=OUT/'bobrov_valerii_exam_report.pdf'
SimpleDocTemplate(str(path),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=38,bottomMargin=43,title='SWEET.TV recommendation: verified experiments',author='Valerii Bobrov').build(story,onFirstPage=footer,onLaterPages=footer)
(D/'report-source.txt').write_text('\n\n'.join(textlog),encoding='utf-8')
reader=PdfReader(path);print('PDF:',path,'pages:',len(reader.pages));print([(i+1,len(p.extract_text())) for i,p in enumerate(reader.pages)])
