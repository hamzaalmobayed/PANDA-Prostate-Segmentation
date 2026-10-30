"""Reference data of the PANDA manuscript (revision of 30 Oct 2026).

One structured record per reference. make_bibliography.py writes, from these
records, the Vancouver-style thebibliography of the manuscript (in order of
first citation) and references.bib. Details were checked against Crossref and
DataCite (PANDA_reference_check.pdf and data/refs/*.json of this revision).
Titles are in sentence case, journal names in NLM abbreviation, no issue
numbers, and every reference with a DOI ends with it.
"""

R = {}


def ref(key, **kw):
    R[key] = kw


# ---- journal articles: authors (Vancouver), title, journal, year, volume, pages, doi
ref("bib1", type="article", authors="Bray F, Laversanne M, Sung H, Ferlay J, Siegel RL, Soerjomataram I, et al.",
    title="Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries",
    journal="CA Cancer J Clin", year=2024, volume="74", pages="229--263", doi="10.3322/caac.21834")
ref("bib3", type="article", authors="Epstein JI, Egevad L, Amin MB, Delahunt B, Srigley JR, Humphrey PA.",
    title="The 2014 International Society of Urological Pathology (ISUP) consensus conference on Gleason grading of prostatic carcinoma: definition of grading patterns and proposal for a new grading system",
    journal="Am J Surg Pathol", year=2016, volume="40", pages="244--252", doi="10.1097/PAS.0000000000000530")
ref("bib4", type="article", authors="Epstein JI, Amin MB, Reuter VE, Humphrey PA.",
    title="Contemporary Gleason grading of prostatic carcinoma: an update with discussion on practical issues to implement the 2014 International Society of Urological Pathology (ISUP) consensus conference on Gleason grading of prostatic carcinoma",
    journal="Am J Surg Pathol", year=2017, volume="41", pages="e1--e7", doi="10.1097/PAS.0000000000000820")
ref("bib5", type="article", authors="Humphrey PA.", title="Gleason grading and prognostic factors in carcinoma of the prostate",
    journal="Mod Pathol", year=2004, volume="17", pages="292--306", doi="10.1038/modpathol.3800054")
ref("bib6", type="article", authors="Ozkan TA, Eruyar AT, Cebeci OO, Memik O, Ozcan L, Kuskonmaz I.",
    title="Interobserver variability in Gleason histological grading of prostate cancer",
    journal="Scand J Urol", year=2016, volume="50", pages="420--424", doi="10.1080/21681805.2016.1206619")
ref("bib7", type="article", authors="Egevad L, Ahmad AS, Algaba F, Berney DM, Boccon-Gibod L, Comp\\'erat E, et al.",
    title="Standardization of Gleason grading among 337 European pathologists",
    journal="Histopathology", year=2013, volume="62", pages="247--256", doi="10.1111/his.12008")
ref("bib8", type="article", authors="Pantanowitz L, Valenstein PN, Evans AJ, Kaplan KJ, Pfeifer JD, Wilbur DC, et al.",
    title="Review of the current state of whole slide imaging in pathology", journal="J Pathol Inform", year=2011, volume="2",
    pages="36", doi="10.4103/2153-3539.83746")
ref("bib9", type="article", authors="Madabhushi A, Lee G.", title="Image analysis and machine learning in digital pathology: challenges and opportunities",
    journal="Med Image Anal", year=2016, volume="33", pages="170--175", doi="10.1016/j.media.2016.06.037")
ref("bib10", type="article", authors="Bulten W, Pinckaers H, van Boven H, Vink R, de Bel T, van Ginneken B, et al.",
    title="Automated deep-learning system for Gleason grading of prostate cancer using biopsies: a diagnostic study",
    journal="Lancet Oncol", year=2020, volume="21", pages="233--241", doi="10.1016/S1470-2045(19)30739-9")
ref("bib11", type="article", authors="Bulten W, Kartasalo K, Chen PHC, Str\\\"om P, Pinckaers H, Nagpal K, et al.",
    title="Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge",
    journal="Nat Med", year=2022, volume="28", pages="154--163", doi="10.1038/s41591-021-01620-2")
ref("bib12", type="article", authors="Nagpal K, Foote D, Liu Y, Chen PHC, Wulczyn E, Tan F, et al.",
    title="Development and validation of a deep learning algorithm for improving Gleason scoring of prostate cancer",
    journal="NPJ Digit Med", year=2019, volume="2", pages="48", doi="10.1038/s41746-019-0112-2")
ref("bib13", type="article", authors="Str\\\"om P, Kartasalo K, Olsson H, Solorzano L, Delahunt B, Berney DM, et al.",
    title="Artificial intelligence for diagnosis and grading of prostate cancer in biopsies: a population-based, diagnostic study",
    journal="Lancet Oncol", year=2020, volume="21", pages="222--232", doi="10.1016/S1470-2045(19)30738-7")
ref("bib14", type="article", authors="Tolkach Y, Dohmg\\\"orgen T, Toma M, Kristiansen G.", title="High-accuracy prostate cancer pathology using deep learning",
    journal="Nat Mach Intell", year=2020, volume="2", pages="411--418", doi="10.1038/s42256-020-0200-7")
ref("bib15", type="article", authors="Gleason DF, Mellinger GT.",
    title="Prediction of prognosis for prostatic adenocarcinoma by combined histological grading and clinical staging",
    journal="J Urol", year=1974, volume="111", pages="58--64", doi="10.1016/S0022-5347(17)59889-4")
ref("bibSchmitz", type="article", authors="Schmitz R, Madesta F, Nielsen M, Krause J, Steurer S, Werner R, et al.",
    title="Multi-scale fully convolutional neural networks for histopathology image segmentation: from nuclear aberrations to the global tissue architecture",
    journal="Med Image Anal", year=2021, volume="70", pages="101996", doi="10.1016/j.media.2021.101996")
ref("bib17", type="article", authors="van Rijthoven M, Balkenhol M, Sili\\c{n}a K, van der Laak J, Ciompi F.",
    title="HookNet: multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images",
    journal="Med Image Anal", year=2021, volume="68", pages="101890", doi="10.1016/j.media.2020.101890")
ref("bib18", type="inproceedings", authors="Tokunaga H, Teramoto Y, Yoshizawa A, Bise R.",
    title="Adaptive weighting multi-field-of-view CNN for semantic segmentation in pathology",
    booktitle="2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", publisher="IEEE", year=2019,
    pages="12589--12598", doi="10.1109/CVPR.2019.01288")
ref("bibHo", type="article", authors="Ho DJ, Yarlagadda DVK, D'Alfonso TM, Hanna MG, Grabenstetter A, Ntiamoah P, et al.",
    title="Deep multi-magnification networks for multi-class breast cancer image segmentation",
    journal="Comput Med Imaging Graph", year=2021, volume="88", pages="101866", doi="10.1016/j.compmedimag.2021.101866")
ref("bib19", type="article", authors="Janowczyk A, Madabhushi A.",
    title="Deep learning for digital pathology image analysis: a comprehensive tutorial with selected use cases",
    journal="J Pathol Inform", year=2016, volume="7", pages="29", doi="10.4103/2153-3539.186902")
ref("bib20", type="article", authors="Tellez D, Litjens G, B\\'andi P, Bulten W, Bokhorst JM, Ciompi F, et al.",
    title="Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology",
    journal="Med Image Anal", year=2019, volume="58", pages="101544", doi="10.1016/j.media.2019.101544")
ref("bib21", type="inproceedings", authors="Ciompi F, Geessink O, Bejnordi BE, Silva de Souza G, Baidoshvili A, Litjens G, et al.",
    title="The importance of stain normalization in colorectal tissue classification with convolutional networks",
    booktitle="2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017)", publisher="IEEE", year=2017,
    pages="160--163", doi="10.1109/ISBI.2017.7950492")
ref("bib22", type="inproceedings", authors="Macenko M, Niethammer M, Marron JS, Borland D, Woosley JT, Guan X, et al.",
    title="A method for normalizing histology slides for quantitative analysis",
    booktitle="2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro", publisher="IEEE", year=2009,
    pages="1107--1110", doi="10.1109/ISBI.2009.5193250")
ref("bib23", type="article", authors="Vahadane A, Peng T, Sethi A, Albarqouni S, Wang L, Baust M, et al.",
    title="Structure-preserving color normalization and sparse stain separation for histological images",
    journal="IEEE Trans Med Imaging", year=2016, volume="35", pages="1962--1971", doi="10.1109/TMI.2016.2529665")
ref("bib24", type="article", authors="Shorten C, Khoshgoftaar TM.", title="A survey on image data augmentation for deep learning",
    journal="J Big Data", year=2019, volume="6", pages="60", doi="10.1186/s40537-019-0197-0")
ref("bib25", type="article", authors="Ciga O, Xu T, Nofech-Mozes S, Noy S, Lu FI, Martel AL.",
    title="Overcoming the limitations of patch-based learning to detect cancer in whole slide images",
    journal="Sci Rep", year=2021, volume="11", pages="8894", doi="10.1038/s41598-021-88494-z")
ref("bib26", type="article", authors="Srinidhi CL, Ciga O, Martel AL.", title="Deep neural network models for computational histopathology: a survey",
    journal="Med Image Anal", year=2021, volume="67", pages="101813", doi="10.1016/j.media.2020.101813")
ref("bib63", type="inproceedings", authors="Bussola N, Marcolini A, Maggio V, Jurman G, Furlanello C.",
    title="AI slipping on tiles: data leakage in digital pathology",
    booktitle="Pattern Recognition. ICPR International Workshops and Challenges. Lecture Notes in Computer Science",
    publisher="Springer", year=2021, pages="167--182", doi="10.1007/978-3-030-68763-2_13")
ref("bib27", type="article", authors="Roberts M, Driggs D, Thorpe M, Gilbey J, Yeung M, Ursprung S, et al.",
    title="Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans",
    journal="Nat Mach Intell", year=2021, volume="3", pages="199--217", doi="10.1038/s42256-021-00307-0")
ref("bib28", type="article", authors="Maier-Hein L, Eisenmann M, Reinke A, Onogur S, Stankovic M, Scholz P, et al.",
    title="Why rankings of biomedical image analysis competitions should be interpreted with care",
    journal="Nat Commun", year=2018, volume="9", pages="5217", doi="10.1038/s41467-018-07619-7")
ref("bib64", type="article", authors="Nir G, Hor S, Karimi D, Fazli L, Skinnider BF, Tavassoli P, et al.",
    title="Automatic grading of prostate cancer in digitized histopathology images: learning from multiple experts",
    journal="Med Image Anal", year=2018, volume="50", pages="167--180", doi="10.1016/j.media.2018.09.005")
ref("bib65", type="misc", authors="", title="Automated Gleason Grading Challenge 2022 (AGGC22): challenge dataset and annotation protocol",
    howpublished="Grand Challenge", year=2022, url="https://aggc22.grand-challenge.org/Data/")
ref("bibBulten2019", type="article", authors="Bulten W, B\\'andi P, Hoven J, van de Loo R, Lotz J, Weiss N, et al.",
    title="Epithelium segmentation using deep learning in H\\&E-stained prostate specimens with immunohistochemistry as reference standard",
    journal="Sci Rep", year=2019, volume="9", pages="864", doi="10.1038/s41598-018-37257-4")
ref("bib45", type="article", authors="Silva-Rodr\\'iguez J, Colomer A, Sales MA, Molina R, Naranjo V.",
    title="Going deeper through the Gleason scoring scale: an automatic end-to-end system for histology prostate grading and cribriform pattern detection",
    journal="Comput Methods Programs Biomed", year=2020, volume="195", pages="105637", doi="10.1016/j.cmpb.2020.105637")
ref("bibAnklin", type="inproceedings", authors="Anklin V, Pati P, Jaume G, Bozorgtabar B, Foncubierta-Rodriguez A, Thiran JP, et al.",
    title="Learning whole-slide segmentation from inexact and incomplete labels using tissue graphs",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021. Lecture Notes in Computer Science",
    publisher="Springer", year=2021, pages="636--646", doi="10.1007/978-3-030-87196-3_59")
ref("bib59", type="article", authors="Chen CM, Huang YS, Fang PW, Liang CW, Chang RF.",
    title="A computer-aided diagnosis system for differentiation and delineation of malignant regions on whole-slide prostate histopathology image using spatial statistics and multidimensional DenseNet",
    journal="Med Phys", year=2020, volume="47", pages="1021--1033", doi="10.1002/mp.13964")
ref("bib62", type="article", authors="Damkliang K, Thongsuksai P, Wongsirichot T, Kayasut K.",
    title="Detection and score grading for prostate adenocarcinoma using semantic segmentation",
    journal="PLoS One", year=2025, volume="20", pages="e0331613", doi="10.1371/journal.pone.0331613")
ref("bib57", type="article", authors="Chen J, Mei J, Li X, Lu Y, Yu Q, Wei Q, et al.",
    title="TransUNet: rethinking the U-Net architecture design for medical image segmentation through the lens of transformers",
    journal="Med Image Anal", year=2024, volume="97", pages="103280", doi="10.1016/j.media.2024.103280")
ref("bib58", type="inproceedings", authors="Cao H, Wang Y, Chen J, Jiang D, Zhang X, Tian Q, et al.",
    title="Swin-Unet: Unet-like pure transformer for medical image segmentation",
    booktitle="Computer Vision -- ECCV 2022 Workshops. Lecture Notes in Computer Science", publisher="Springer",
    year=2023, pages="205--218", doi="10.1007/978-3-031-25066-8_9")
ref("bib48", type="article", authors="Chen RJ, Ding T, Lu MY, Williamson DFK, Jaume G, Song AH, et al.",
    title="Towards a general-purpose foundation model for computational pathology", journal="Nat Med", year=2024, volume="30",
    pages="850--862", doi="10.1038/s41591-024-02857-3")
ref("bib49", type="article", authors="Vorontsov E, Bozkurt A, Casson A, Shaikovski G, Zelechowski M, Severson K, et al.",
    title="A foundation model for clinical-grade computational pathology and rare cancers detection", journal="Nat Med", year=2024,
    volume="30", pages="2924--2935", doi="10.1038/s41591-024-03141-0")
ref("bib50", type="article", authors="Xu H, Usuyama N, Bagga J, Zhang S, Rao R, Naumann T, et al.",
    title="A whole-slide foundation model for digital pathology from real-world data", journal="Nature", year=2024, volume="630",
    pages="181--188", doi="10.1038/s41586-024-07441-w")
ref("bib29", type="inproceedings", authors="Ronneberger O, Fischer P, Brox T.", title="U-Net: convolutional networks for biomedical image segmentation",
    booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015. Lecture Notes in Computer Science",
    publisher="Springer", year=2015, pages="234--241", doi="10.1007/978-3-319-24574-4_28")
ref("bib30", type="inproceedings", authors="Tan M, Le Q.", title="EfficientNet: rethinking model scaling for convolutional neural networks",
    booktitle="Proceedings of the 36th International Conference on Machine Learning (PMLR 97)",
    publisher="PMLR", year=2019, pages="6105--6114")
ref("bibPANDAdesign", type="misc", authors="Bulten W, Litjens G, Pinckaers H, Str\\\"om P, Eklund M, Kartasalo K, et al.",
    title="The PANDA challenge: Prostate cANcer graDe Assessment using the Gleason grading system [challenge design document]",
    howpublished="Zenodo", year=2020, doi="10.5281/zenodo.3715938")
ref("bib31", type="article", authors="Goode A, Gilbert B, Harkes J, Jukic D, Satyanarayanan M.",
    title="OpenSlide: a vendor-neutral software foundation for digital pathology", journal="J Pathol Inform", year=2013, volume="4",
    pages="27", doi="10.4103/2153-3539.119005")
ref("bib32", type="article", authors="Pocock J, Graham S, Vu QD, Jahanifar M, Deshpande S, Hadjigeorghiou G, et al.",
    title="TIAToolbox as an end-to-end library for advanced tissue image analytics", journal="Commun Med (Lond)", year=2022,
    volume="2", pages="120", doi="10.1038/s43856-022-00186-5")
ref("bib34", type="article", authors="Lloyd SP.", title="Least squares quantization in PCM", journal="IEEE Trans Inf Theory", year=1982,
    volume="28", pages="129--137", doi="10.1109/TIT.1982.1056489")
ref("bib36", type="inproceedings", authors="Loshchilov I, Hutter F.", title="Decoupled weight decay regularization",
    booktitle="7th International Conference on Learning Representations (ICLR 2019)", publisher="OpenReview.net", year=2019, pages="")
ref("bib37", type="inproceedings", authors="Micikevicius P, Narang S, Alben J, Diamos G, Elsen E, Garcia D, et al.", title="Mixed precision training",
    booktitle="6th International Conference on Learning Representations (ICLR 2018)", publisher="OpenReview.net", year=2018, pages="")
ref("bib38", type="article", authors="Dice LR.", title="Measures of the amount of ecologic association between species", journal="Ecology",
    year=1945, volume="26", pages="297--302", doi="10.2307/1932409")
ref("bib39", type="article", authors="Reinke A, Tizabi MD, Baumgartner M, Eisenmann M, Heckmann-N\\\"otzel D, Kavur AE, et al.",
    title="Understanding metric-related pitfalls in image analysis validation", journal="Nat Methods", year=2024, volume="21",
    pages="182--194", doi="10.1038/s41592-023-02150-0")
ref("bibMetricsReloaded", type="article", authors="Maier-Hein L, Reinke A, Godau P, Tizabi MD, Buettner F, Christodoulou E, et al.",
    title="Metrics reloaded: recommendations for image analysis validation", journal="Nat Methods", year=2024, volume="21",
    pages="195--212", doi="10.1038/s41592-023-02151-z")
ref("bib40", type="book", authors="Efron B, Tibshirani RJ.", title="An introduction to the bootstrap", publisher="Chapman \\& Hall/CRC",
    year=1993, doi="10.1007/978-1-4899-4541-9")
ref("bib41", type="article", authors="Holm S.", title="A simple sequentially rejective multiple test procedure", journal="Scand J Stat",
    year=1979, volume="6", pages="65--70")
ref("bib42", type="article", authors="Taha AA, Hanbury A.", title="Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool",
    journal="BMC Med Imaging", year=2015, volume="15", pages="29", doi="10.1186/s12880-015-0068-x")
ref("bib43", type="article", authors="Nikolov S, Blackwell S, Zverovitch A, Mendes R, Livne M, De Fauw J, et al.",
    title="Clinically applicable segmentation of head and neck anatomy for radiotherapy: deep learning algorithm development and validation study",
    journal="J Med Internet Res", year=2021, volume="23", pages="e26151", doi="10.2196/26151")
ref("bib44", type="misc", authors="Silva-Rodr\\'iguez J.",
    title="SICAPv2 -- prostate whole slide images with Gleason grades annotations [dataset]. Version 2", howpublished="Mendeley Data",
    year=2020, doi="10.17632/9xxm58dvs3.2")
ref("bib46", type="article", authors="Calapaqu\\'i Ter\\'an AK, Gonz\\'alez Bernad AA, Cobo Cano M, S\\'anchez Magdaleno L, Marcos Gonz\\'alez S, Delgado Bolton RC, et al.",
    title="PRECISE: benchmarking digital pathology with expert-annotated contiguous IHC--H\\&E serial prostate sections",
    journal="Mach Learn Biomed Imaging", year=2026, volume="2026", pages="825--833", doi="10.59275/j.melba.2026-g657")
ref("bibPRECISEdata", type="misc", authors="Calapaqu\\'i Ter\\'an AK, Gonz\\'alez Bernad AA, S\\'anchez Magdaleno L, Marcos Gonz\\'alez S, Delgado Bolton RC, Cobo Cano M, et al.",
    title="PRECISE -- PRostate Expert-annotated Contiguous IHC--H\\&E Serial sEctions [dataset]", howpublished="Zenodo", year=2026,
    doi="10.5281/zenodo.20721779")
ref("bibKarram", type="article", authors="Karram S, Trock BJ, Netto GJ, Epstein JI.",
    title="Should intervening benign tissue be included in the measurement of discontinuous foci of cancer on prostate needle biopsy? Correlation with radical prostatectomy findings",
    journal="Am J Surg Pathol", year=2011, volume="35", pages="1351--1355", doi="10.1097/PAS.0b013e3182217b79")
ref("bib47", type="article", authors="Tolkach Y, Ovtcharov V, Pryalukhin A, Eich ML, Gaisa NT, Braun M, et al.",
    title="An international multi-institutional validation study of the algorithm for prostate cancer detection and Gleason grading",
    journal="NPJ Precis Oncol", year=2023, volume="7", pages="77", doi="10.1038/s41698-023-00424-6")
ref("bib51", type="article", authors="Topol EJ.", title="High-performance medicine: the convergence of human and artificial intelligence",
    journal="Nat Med", year=2019, volume="25", pages="44--56", doi="10.1038/s41591-018-0300-7")
ref("bib52", type="article", authors="Nagendran M, Chen Y, Lovejoy CA, Gordon AC, Komorowski M, Harvey H, et al.",
    title="Artificial intelligence versus clinicians: systematic review of design, reporting standards, and claims of deep learning studies",
    journal="BMJ", year=2020, volume="368", pages="m689", doi="10.1136/bmj.m689")
