# src/verdicts.py — Phase 6 & Final Build: Disease name, safe-to-consume lookup, treatment, storage & OOD guardrails
import numpy as np

HONESTY_LINE = (
    "Visual estimate from appearance only — not a laboratory microbiological food-safety test."
)

# ─────────────────────────────────────────────────────────────
# PlantVillage classes (38 classes from mohanty/PlantVillage)
# Format: "ClassName": (status, disease_name, safe_bool, verdict_text, note, treatment, prevention)
# ─────────────────────────────────────────────────────────────
_PLANT_VERDICTS = {
    # Apple
    "Apple___Apple_scab": (
        "Apple Leaf — Apple Scab", "Apple Scab (Venturia inaequalis)", False, "Not safe — discard affected fruit",
        "Visible fungal infection on apple leaf; fruit from severely infected trees may have scab lesions.",
        "Apply organic sulfur or copper-based fungicide; prune and destroy fallen infected leaves.",
        "Plant scab-resistant cultivars, maintain open canopy airflow, and avoid overhead irrigation."
    ),
    "Apple___Black_rot": (
        "Apple Leaf — Black Rot", "Black Rot (Botryosphaeria obtusa)", False, "Not safe — discard",
        "Fungal/bacterial rot affecting fruit, leaves, and bark.",
        "Prune dead wood and mummified fruit; apply captan or sulfur fungicides during bloom.",
        "Sanitize pruning tools with 70% alcohol and remove nearby wild brambles or dead limbs."
    ),
    "Apple___Cedar_apple_rust": (
        "Apple Leaf — Cedar Apple Rust", "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)", False, "Not safe — discard",
        "Bright orange-yellow fungal rust lesions on apple foliage.",
        "Apply myclobutanil or copper fungicide at pink bud and petal fall stages.",
        "Remove nearby Eastern red cedar / juniper hosts within a 300m radius if possible."
    ),
    "Apple___healthy": (
        "Healthy Apple Leaf", "None", True, "Likely safe — Healthy Plant",
        "Apple foliage appears vibrant and free of visible lesions or pathogens.",
        "No treatment needed; continue standard orchard nutrition and watering.",
        "Inspect canopy bi-weekly and maintain balanced soil pH (6.0–6.5)."
    ),
    # Blueberry
    "Blueberry___healthy": (
        "Healthy Blueberry Leaf", "None", True, "Likely safe — Healthy Plant",
        "Blueberry leaf looks healthy with normal coloration.",
        "No intervention needed.",
        "Maintain acidic soil (pH 4.5–5.5) with pine bark mulch and regular drip irrigation."
    ),
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": (
        "Cherry Leaf — Powdery Mildew", "Powdery Mildew (Podosphaera clandestina)", False, "Not safe — discard affected parts",
        "White powdery fungal mycelium coating cherry leaf surfaces.",
        "Spray potassium bicarbonate, neem oil, or sulfur fungicide at first sign of white patches.",
        "Prune summer water sprouts to increase sunlight penetration and reduce canopy humidity."
    ),
    "Cherry_(including_sour)___healthy": (
        "Healthy Cherry Leaf", "None", True, "Likely safe — Healthy Plant",
        "Cherry leaf looks healthy and free of fungal or bacterial spots.",
        "No treatment required.",
        "Ensure good soil drainage and annual dormant pruning."
    ),
    # Corn / Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Corn Leaf — Gray Leaf Spot", "Gray Leaf Spot (Cercospora zeae-maydis)", False, "Not safe — discard infected foliage",
        "Rectangular tan-to-gray fungal lesions bounded by leaf veins.",
        "Apply strobilurin or triazole foliar fungicides if lesions reach the ear leaf before tasseling.",
        "Practice 2-year crop rotation away from maize and till crop debris after harvest."
    ),
    "Corn_(maize)___Common_rust_": (
        "Corn Leaf — Common Rust", "Common Rust (Puccinia sorghi)", False, "Not safe — discard infected foliage",
        "Cinnamon-brown pustules erupting on both upper and lower leaf surfaces.",
        "Apply foliar fungicide if pustules cover >5% of leaf area prior to silking.",
        "Plant rust-resistant hybrid seeds and avoid late planting dates."
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Corn Leaf — Northern Leaf Blight", "Northern Corn Leaf Blight (Exserohilum turcicum)", False, "Not safe — discard",
        "Large cigar-shaped grayish-green necrotic lesions on maize leaves.",
        "Apply propiconazole or azoxystrobin fungicide at early lesion appearance.",
        "Use hybrids with Ht resistance genes and manage surface crop residue."
    ),
    "Corn_(maize)___healthy": (
        "Healthy Corn (Maize) Leaf", "None", True, "Likely safe — Healthy Plant",
        "Maize leaf shows uniform green chlorophyll with no rust or blight.",
        "No treatment needed.",
        "Provide adequate nitrogen fertilization and monitor moisture during tasseling."
    ),
    # Grape
    "Grape___Black_rot": (
        "Grape Leaf — Black Rot", "Grape Black Rot (Guignardia bidwellii)", False, "Not safe — discard",
        "Reddish-brown circular foliar spots and shriveled black mummified berries.",
        "Remove all mummified clusters; apply mancozeb or myclobutanil from bud break through veraison.",
        "Ensure full sun exposure on fruit zones via leaf pulling and trellis training."
    ),
    "Grape___Esca_(Black_Measles)": (
        "Grape Leaf — Esca (Black Measles)", "Esca / Black Measles Complex", False, "Not safe — discard",
        "Interveinal 'tiger-stripe' chlorosis and necrosis caused by trunk fungal pathogens.",
        "Excise infected cordons below stained wood; seal pruning wounds with boron paste or trichoderma.",
        "Prune only during dry weather in late winter to minimize spore entry into vascular tissue."
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape Leaf — Isariopsis Leaf Blight", "Isariopsis Leaf Spot (Pseudocercospora vitis)", False, "Not safe — discard",
        "Irregular dark-brown angular lesions on older grape leaves.",
        "Apply copper oxychloride or captan sprays; collect and burn fallen foliage.",
        "Improve vineyard floor drainage and control weeds under trellises."
    ),
    "Grape___healthy": (
        "Healthy Grape Leaf", "None", True, "Likely safe — Healthy Plant",
        "Grape leaf looks healthy and vigorous.",
        "No treatment required.",
        "Maintain balanced potassium/magnesium nutrition and canopy ventilation."
    ),
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Orange Leaf — Citrus Greening (HLB)", "Huanglongbing (Candidatus Liberibacter)", False, "Not safe — fruit bitter/unfit",
        "Asymmetrical blotchy mottling on leaf; systemic bacterial vascular infection.",
        "No known cure once infected; control Asian citrus psyllid vector with horticultural oils or systemic insecticides and remove severely declining trees.",
        "Use certified HLB-free nursery stock and deploy yellow sticky traps to monitor psyllids."
    ),
    # Peach
    "Peach___Bacterial_spot": (
        "Peach Leaf — Bacterial Spot", "Bacterial Spot (Xanthomonas arboricola)", False, "Not safe — discard",
        "Angular water-soaked purple-black lesions that drop out leaving 'shot-holes'.",
        "Apply fixed copper sprays at leaf drop and dormant bud swell; oxytetracycline during shuck split.",
        "Plant windbreaks to reduce sandblasting leaf injury and avoid excessive nitrogen."
    ),
    "Peach___healthy": (
        "Healthy Peach Leaf", "None", True, "Likely safe — Healthy Plant",
        "Peach leaf looks healthy with smooth margins.",
        "No treatment needed.",
        "Perform annual open-center pruning and dormant oil sprays."
    ),
    # Pepper / Bell
    "Pepper,_bell___Bacterial_spot": (
        "Bell Pepper Leaf — Bacterial Spot", "Bacterial Leaf Spot (Xanthomonas campestris)", False, "Not safe — discard",
        "Dark greasy necrotic spots with yellow halos on bell pepper foliage.",
        "Spray copper hydroxide tank-mixed with mancozeb every 7–10 days in humid weather.",
        "Use hot-water treated certified seed and avoid working in fields when foliage is wet."
    ),
    "Pepper,_bell___healthy": (
        "Healthy Bell Pepper Leaf", "None", True, "Likely safe — Healthy Plant",
        "Bell pepper leaf looks healthy and glossy.",
        "No treatment needed.",
        "Mulch beds to prevent soil splash and maintain consistent calcium availability."
    ),
    # Potato
    "Potato___Early_blight": (
        "Potato Leaf — Early Blight", "Early Blight (Alternaria solani)", False, "Not safe — discard affected parts",
        "Dark brown concentric 'target-board' rings on lower senescing leaves.",
        "Apply chlorothalonil, azoxystrobin, or copper fungicide; remove heavily spotted lower leaves.",
        "Hilling soil over tubers prevents spore wash-down; rotate with non-solanaceous crops for 3 years."
    ),
    "Potato___Late_blight": (
        "Potato Leaf — Late Blight", "Late Blight (Phytophthora infestans)", False, "Not safe — do not consume",
        "Aggressive water-soaked dark blotches with white cottony sporulation on leaf undersides.",
        "Immediately rogue and bag infected haulms; apply targeted oomycete fungicides (mefenoxam/cymoxanil).",
        "Destroy all volunteer potatoes and cull piles; plant certified disease-free seed tubers."
    ),
    "Potato___healthy": (
        "Healthy Potato Leaf", "None", True, "Likely safe — Healthy Plant",
        "Potato plant foliage looks healthy and turgid.",
        "No treatment needed.",
        "Hill soil around stems and irrigate early in the day so foliage dries rapidly."
    ),
    # Raspberry
    "Raspberry___healthy": (
        "Healthy Raspberry Leaf", "None", True, "Likely safe — Healthy Plant",
        "Raspberry leaf looks healthy.",
        "No treatment required.",
        "Prune out spent floricanes immediately after fruiting to keep rows airy."
    ),
    # Soybean
    "Soybean___healthy": (
        "Healthy Soybean Leaf", "None", True, "Likely safe — Healthy Plant",
        "Soybean plant foliage looks healthy.",
        "No treatment needed.",
        "Maintain crop rotation with corn/small grains and monitor for defoliating insects."
    ),
    # Squash
    "Squash___Powdery_mildew": (
        "Squash Leaf — Powdery Mildew", "Powdery Mildew (Podosphaera xanthii)", False, "Not safe — discard infected leaf",
        "Talcum-like white fungal growth covering squash leaf lamina.",
        "Spray neem oil, potassium bicarbonate, or wettable sulfur (when temps < 29°C) every 7 days.",
        "Space hills widely for air circulation and water at the root zone only."
    ),
    # Strawberry
    "Strawberry___Leaf_scorch": (
        "Strawberry Leaf — Leaf Scorch", "Leaf Scorch (Diplocarpon earliana)", False, "Not safe — discard",
        "Numerous irregular purplish blotches coalescing to make the leaf look burned/scorched.",
        "Clip and remove infected leaves after harvest renovation; apply captan or copper sprays.",
        "Renovate matted rows annually, renew mulch, and use drip tape instead of sprinklers."
    ),
    "Strawberry___healthy": (
        "Healthy Strawberry Leaf", "None", True, "Likely safe — Healthy Plant",
        "Strawberry plant foliage looks bright green and healthy.",
        "No treatment needed.",
        "Keep crowns above soil line and mulch with clean straw to keep berries off damp soil."
    ),
    # Tomato
    "Tomato___Bacterial_spot": (
        "Tomato Leaf — Bacterial Spot", "Bacterial Spot (Xanthomonas spp.)", False, "Not safe — discard",
        "Small dark brown greasy lesions on tomato leaves and raised scabs on fruit.",
        "Apply copper-based bactericides; prune lower leaves to reduce humidity.",
        "Avoid overhead watering, sanitize stakes/cages with 10% bleach, and rotate crops."
    ),
    "Tomato___Early_blight": (
        "Tomato Leaf — Early Blight", "Early Blight (Alternaria linariae)", False, "Not safe — discard",
        "Bullseye concentric brown rings surrounded by yellow chlorotic tissue on older leaves.",
        "Prune bottom 12 inches of foliage; spray copper fungicide, serenade (Bacillus subtilis), or chlorothalonil.",
        "Apply 2–3 inches of organic mulch to block soil-borne spores from splashing onto lower leaves."
    ),
    "Tomato___Late_blight": (
        "Tomato Leaf — Late Blight", "Late Blight (Phytophthora infestans)", False, "Not safe — do not consume",
        "Greasy gray-black expanding lesions on leaves and stems; causes rapid firm brown fruit rot.",
        "Uproot and seal severely infected plants in plastic bags (do not compost); spray copper/mancozeb protectively on adjacent plants.",
        "Plant resistant varieties (e.g., 'Defiant', 'Mountain Magic') and keep foliage dry under high tunnels."
    ),
    "Tomato___Leaf_Mold": (
        "Tomato Leaf — Leaf Mold", "Tomato Leaf Mold (Passalora fulva)", False, "Not safe — discard",
        "Pale yellow spots on upper leaf surface with olive-green velvety mold underneath.",
        "Ventilate greenhouse/tunnel immediately to drop relative humidity below 85%; apply copper or calcium chloride sprays.",
        "Prune dense sucker growth and space plants at least 24–36 inches apart."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Tomato Leaf — Septoria Leaf Spot", "Septoria Leaf Spot (Septoria lycopersici)", False, "Not safe — discard",
        "Numerous small circular spots with dark brown margins and tan centers containing black pycnidia.",
        "Remove infected lower leaves; apply chlorothalonil, mancozeb, or organic copper fungicide.",
        "Eliminate nightshade weeds (horsenettle, jimsonweed) and rotate away from solanaceous crops."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato Leaf — Two-Spotted Spider Mites", "Spider Mite Infestation (Tetranychus urticae)", False, "Not safe — wash/discard damaged parts",
        "Yellow stippling/flecking on leaves accompanied by fine silk webbing on undersides.",
        "Wash undersides of leaves with a strong water jet; apply insecticidal soap, neem oil, or release predatory mites (Phytoseiulus persimilis).",
        "Avoid dusty conditions and water-stressed plants, which accelerate mite reproduction in hot weather."
    ),
    "Tomato___Target_Spot": (
        "Tomato Leaf — Target Spot", "Target Spot (Corynespora cassiicola)", False, "Not safe — discard",
        "Brown necrotic lesions with subtle concentric rings and yellow margins.",
        "Apply azoxystrobin, boscalid, or chlorothalonil fungicides; remove senescent lower leaves.",
        "Improve canopy aeration through staking/trellising and avoid excessive late-day irrigation."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato Leaf — Yellow Leaf Curl Virus", "Tomato Yellow Leaf Curl Virus (TYLCV)", False, "Not safe — stunted/unfit",
        "Upward curling, cupping, and interveinal yellowing of young leaves with stunted plant growth.",
        "Remove and destroy infected plants immediately; manage silverleaf whitefly vectors (Bemisia tabaci) with neem oil or imidacloprid.",
        "Use reflective silver plastic mulch and fine insect exclusion netting (50-mesh) in nurseries."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato Leaf — Tomato Mosaic Virus", "Tomato Mosaic Virus (ToMV)", False, "Not safe — discard",
        "Light and dark green mosaic mottling, fern-like leaf distortion, and internal browning of fruit.",
        "No chemical cure for viral infections; rogue infected plants and wash hands/tools with milk protein or trisodium phosphate (TSP).",
        "Do not handle tobacco products before touching solanaceous plants; plant ToMV-resistant cultivars."
    ),
    "Tomato___healthy": (
        "Healthy Tomato Leaf", "None", True, "Likely safe — Healthy Plant",
        "Tomato foliage is deep green, turgid, and free of spots or curling.",
        "No treatment needed; continue regular staking, pruning suckers, and consistent watering.",
        "Maintain even soil moisture to prevent blossom-end rot and mulch around base."
    ),
}

# ─────────────────────────────────────────────────────────────
# Fresh / Rotten Fruit & Vegetable classes (70+ classes)
# ─────────────────────────────────────────────────────────────
_STORAGE_GUIDE = {
    "apple": ("Store in crisper drawer at 1–4°C (85–90% humidity); keep away from ethylene-sensitive greens.", "Inspect weekly and remove bruised apples immediately ('one bad apple spoils the barrel' via ethylene)."),
    "banana": ("Store at room temperature (13–15°C); hang to prevent bruising. Refrigerate only when fully ripe.", "Wrap crown stems in plastic wrap to slow ethylene ripening by up to 3–5 days."),
    "beetroot": ("Clip leafy tops leaving 1 inch of stem; store unwashed roots in perforated bags in refrigerator for 2–4 weeks.", "Keep cool and moist to prevent shriveling."),
    "bell_pepper": ("Store dry and unwashed in the vegetable crisper at 7–10°C for up to 2 weeks.", "Dry surface moisture before storing to prevent soft rot in the stem cavity."),
    "cabbage": ("Wrap tightly in breathable film in refrigerator crisper (0–2°C) for 3–6 weeks.", "Do not wash until ready to slice; remove outer wilted wrapper leaves."),
    "capsicum": ("Store dry and unwashed in the vegetable crisper at 7–10°C for up to 2 weeks.", "Keep dry to prevent bacterial soft rot around the calyx."),
    "carrot": ("Remove green tops; store in sealed container with a dry paper towel in fridge (0–2°C) for 3–4 weeks.", "Keep away from apples and pears to avoid ethylene-induced bitterness (isocoumarin)."),
    "cauliflower": ("Store stem-side up in a loosely perforated plastic bag in refrigerator (0–2°C) for 1–2 weeks.", "Stem-side up prevents condensation droplets from pooling on the white curds."),
    "cherry": ("Store unwashed in a shallow breathable container at 0–2°C; wash only right before eating.", "Keep stems attached until consumption to prevent fungal entry at the stem scar."),
    "chilli_pepper": ("Store unwashed wrapped in a dry paper towel inside a breathable bag in fridge crisper for 2–3 weeks.", "Can also be sun-dried, frozen whole, or pickled for long-term preservation."),
    "corn": ("Store unhusked in refrigerator (0–2°C) and consume within 2–3 days before sugars convert to starch.", "Blanch and freeze kernels if storing longer than 72 hours."),
    "cucumber": ("Wrap in a paper towel inside a loose bag in the warmer front part of the fridge (10–12°C).", "Avoid temperatures below 7°C to prevent chilling injury (water-soaked pitting)."),
    "eggplant": ("Store in a cool spot (10–12°C) or crisper drawer wrapped in paper towel; use within 4–5 days.", "Sensitive to both cold chilling injury and ethylene gas from bananas/tomatoes."),
    "garlic": ("Store whole bulbs in an open basket in a cool, dark, dry pantry (15–18°C) with good airflow.", "Do not refrigerate whole bulbs, as cold stimulates premature sprouting and rubbery texture."),
    "ginger": ("Store unpeeled rhizomes in a resealable bag (air pressed out) in the crisper drawer for up to 1 month.", "Peel and mince into ice-cube trays for 6-month freezer storage."),
    "grape": ("Store unwashed in a ventilated bag in the coldest part of the refrigerator (0–2°C).", "Snip off any split or moldy berries immediately before refrigerating."),
    "guava": ("Ripen at room temperature until fragrant, then refrigerate at 5–8°C for up to 5 days.", "Handle gently as thin skin bruises easily and invites anthracnose rot."),
    "jujube": ("Store fresh crisp jujubes in a perforated bag in the refrigerator for 1–2 weeks.", "Can be dehydrated into red dates for year-round shelf stability."),
    "kiwi": ("Hold firm kiwis at room temp to ripen; store ripe kiwis in refrigerator for 1–2 weeks.", "Keep firm kiwis away from apples unless you want to accelerate ripening."),
    "lemon": ("Store in a sealed crisper bag in the refrigerator (4–8°C) for up to 3–4 weeks.", "Room temperature storage dries out the rind within 5–7 days."),
    "lettuce": ("Wash, spin thoroughly dry, and store layered with dry paper towels in an airtight container at 1–3°C.", "Replace damp paper towels every 3 days to prevent slimy bacterial decay."),
    "lime": ("Store in the refrigerator crisper drawer in a sealed plastic bag for 3–4 weeks.", "Avoid direct sunlight which yellows the green rind and reduces juice acidity."),
    "mango": ("Ripen at room temp (18–22°C) until yielding to gentle palm pressure; then refrigerate for up to 5 days.", "Never refrigerate green unripe mangoes (<10°C halts ripening permanently)."),
    "okra": ("Store completely dry in a paper bag in the warmer part of the refrigerator (7–10°C) for 2–3 days.", "Moisture causes rapid blackened sliminess; wash only immediately before cooking."),
    "onion": ("Store cured onions in a single layer or mesh bag in a cool, dry, dark, well-ventilated pantry.", "Never store onions next to potatoes—moisture and gases cause both to spoil rapidly."),
    "orange": ("Store in mesh bag in refrigerator crisper (3–8°C) for 3–4 weeks.", "Ensure air circulation; remove any fruit showing white/blue penicillium mold immediately."),
    "papaya": ("Ripen at room temperature until skin is >50% yellow; refrigerate ripe fruit and consume within 3 days.", "Handle carefully to prevent latex staining and bruised soft spots."),
    "peach": ("Ripen stem-side down on counter at room temperature; move to fridge only when fragrant and slightly soft.", "Refrigerating unripe peaches causes mealy, dry flesh texture (chilling injury)."),
    "pear": ("Ripen at room temperature until the neck yields to gentle thumb pressure, then refrigerate (0–2°C).", "Pears ripen from the inside out—check the neck daily so they don't overripen at the core."),
    "peas": ("Store unshelled pods in a perforated bag in the refrigerator crisper (0–2°C) and eat within 3 days.", "Blanch for 90 seconds and shock in ice water before freezing to lock in sweetness."),
    "pineapple": ("Once cut or ripe, store chunks in an airtight glass container in the fridge (4°C) for 4–5 days.", "Whole uncut pineapple can sit on counter for 1–2 days; turn upside down for 30 mins before slicing."),
    "pomegranate": ("Store whole fruit in refrigerator crisper for up to 1–2 months; extracted arils last 5 days in airtight container.", "Arils freeze excellently on a parchment tray for up to 6 months."),
    "potato": ("Store in a cool (7–10°C), dark, ventilated paper or burlap sack.", "Keep strictly in the dark to prevent toxic green solanine chlorophyll synthesis; do not refrigerate raw potatoes."),
    "pumpkin": ("Cure intact pumpkins and store on cardboard/wood (not cold concrete) at 10–15°C in a dry room.", "Once cut, wrap seeded pieces in film and refrigerate for 4–5 days or puree and freeze."),
    "radish": ("Twist off leafy greens immediately; store roots in a damp paper towel inside a sealed bag in the fridge for 2 weeks.", "Leaving greens attached draws moisture out of the root, making it pithy and spongy."),
    "spinach": ("Store unwashed with dry paper towels in a rigid container in the fridge (0–2°C) for 5–7 days.", "Discard any crushed or yellowing leaves immediately to stop bacterial soft rot."),
    "strawberry": ("Store unwashed in a single layer on a dry paper towel in a partially vented container at 1–2°C.", "Optional: brief 1:10 white vinegar/water rinse followed by thorough drying extends life by 4+ days."),
    "sweetpotato": ("Store unwashed in a cool, dark, ventilated basket (13–16°C); never refrigerate raw sweet potatoes.", "Cold fridge temps (<10°C) cause a hard core and off-flavors when cooked."),
    "tomato": ("Store ripe tomatoes stem-scar up at cool room temperature (13–18°C) out of direct sunlight.", "Avoid refrigeration unless overripe, as cold breaks down volatile aroma compounds and makes flesh mealy."),
    "turnip": ("Remove green tops and store roots in a perforated plastic bag in the crisper drawer (0–2°C) for 2–3 weeks.", "Cook greens within 2 days as they wilt much faster than the root."),
    "watermelon": ("Store whole uncut melon at cool room temp (10–15°C); refrigerate cut wedges tightly wrapped at 2–4°C for 3–4 days.", "Wash the outer rind thoroughly with soap and water before slicing to avoid dragging surface bacteria through the flesh."),
}

_ALL_PRODUCE = list(_STORAGE_GUIDE.keys())

_PRODUCE_VERDICTS = {}
for _p in _ALL_PRODUCE:
    nice_p = _p.replace("_", " ").title()
    storage_tip, prev_tip = _STORAGE_GUIDE[_p]
    _PRODUCE_VERDICTS[f"fresh_{_p}"] = (
        f"Fresh {nice_p}",
        "None (Healthy & Fresh)",
        True,
        "Likely safe to consume",
        f"Fresh {nice_p.lower()} with firm texture, normal color, and no visible decay or lesions.",
        f"Wash thoroughly under running water before preparation. {storage_tip}",
        prev_tip
    )
    _PRODUCE_VERDICTS[f"rotten_{_p}"] = (
        f"Rotten {nice_p}",
        f"{nice_p} Spoilage / Microbial Decay",
        False,
        "Not safe — discard immediately",
        f"Visible tissue breakdown, discoloration, fungal mold, or fermentation in {nice_p.lower()} — mycotoxins may diffuse beyond visible spots.",
        f"Discard the affected {nice_p.lower()} in a sealed compost or trash bin; sanitize any cutting boards, knives, or fridge shelves it touched with warm soapy water.",
        f"To prevent future spoilage: {storage_tip} {prev_tip}"
    )

# Merge all verdicts
_ALL_VERDICTS = {**_PLANT_VERDICTS, **_PRODUCE_VERDICTS}

_DEFAULT = (
    "Unknown / Unlisted Specimen",
    "Unverified",
    False,
    "Uncertain — inspect carefully",
    "Class not in standard database; inspect closely for odor, slime, or mold before use.",
    "Consult an agricultural extension officer or food safety specialist if symptoms are present.",
    "Store produce clean, dry, and at appropriate temperatures."
)


def check_ood_guardrail(preds: np.ndarray, conf_threshold: float = 28.0, entropy_threshold: float = 4.15) -> dict:
    """
    Task 4: Out-of-Distribution (OOD) & Low-Confidence Guardrail.
    Evaluates softmax probability vector for low top-1 confidence or high Shannon entropy.
    """
    top_conf = float(np.max(preds)) * 100.0
    eps = 1e-12
    probs = np.clip(preds, eps, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    
    is_ood = (top_conf < conf_threshold) or (entropy > entropy_threshold and top_conf < 45.0)
    return {
        "is_ood": is_ood,
        "top_confidence": top_conf,
        "entropy": entropy,
        "reason": (
            f"Low visual confidence ({top_conf:.1f}%) / high uncertainty (entropy {entropy:.2f}). "
            "The uploaded image may not be a clear plant leaf, fruit, or vegetable, or lighting/focus is obstructed."
            if is_ood else "Valid in-distribution prediction."
        )
    }


def get_verdict(class_name: str, confidence: float = None, preds: np.ndarray = None) -> dict:
    """
    Returns a dict with keys:
      status, disease, safe, verdict, note, treatment, prevention, is_ood
    Falls back gracefully for unknown class names or OOD inputs.
    """
    # Check OOD guardrail if probability vector or low confidence is supplied
    if preds is not None:
        ood_info = check_ood_guardrail(preds)
        if ood_info["is_ood"]:
            return dict(
                status="Unrecognized / Low Confidence Object",
                disease="Uncertain — Out of Distribution or Blurry Image",
                safe=False,
                verdict="Uncertain — Retake Photo",
                note=ood_info["reason"],
                treatment="Please upload a well-lit, focused close-up photo of a plant leaf, fruit, or vegetable against a plain background.",
                prevention="Ensure the camera lens is clean and the specimen fills at least 40% of the frame.",
                is_ood=True,
            )
    elif confidence is not None and confidence < 25.0:
        return dict(
            status="Low Confidence Prediction",
            disease="Uncertain — Low Confidence",
            safe=False,
            verdict="Uncertain — Inspect Manually",
            note=f"Model confidence is only {confidence:.1f}%. The object may not belong to the 70 supported crop/produce classes.",
            treatment="Capture a clearer image with natural lighting and minimal background clutter.",
            prevention="Ensure the target leaf or vegetable is centered and in focus.",
            is_ood=True,
        )

    entry = _ALL_VERDICTS.get(class_name)
    if entry is None:
        normalised = class_name.strip().replace(" ", "_")
        entry = _ALL_VERDICTS.get(normalised)

    if entry is None:
        lc = class_name.lower()
        if "healthy" in lc or lc.startswith("fresh_"):
            entry = (
                "Healthy / Fresh Specimen", "None", True, "Likely safe to consume",
                "Appears healthy or fresh based on surface features.",
                "Wash with clean running water prior to use.",
                "Store in a clean, temperature-controlled environment."
            )
        elif any(k in lc for k in ("rot", "blight", "mold", "mildew", "rust", "scorch", "virus", "spot", "rotten", "decay")):
            entry = (
                "Diseased / Rotten Specimen", "Pathogen or Spoilage Detected", False, "Not safe — discard",
                "Visible disease lesions or tissue decay — do not consume.",
                "Discard affected item and sanitize contact surfaces.",
                "Isolate healthy produce from decaying items immediately."
            )
        else:
            entry = _DEFAULT

    status, disease, safe, verdict, note, treatment, prevention = entry
    return dict(
        status=status,
        disease=disease,
        safe=safe,
        verdict=verdict,
        note=note,
        treatment=treatment,
        prevention=prevention,
        is_ood=False,
    )


if __name__ == "__main__":
    tests = [
        "Tomato___Late_blight",
        "Apple___healthy",
        "fresh_apple",
        "rotten_apple",
        "rotten_tomato",
        "fresh_carrot",
    ]
    for cls in tests:
        v = get_verdict(cls)
        print(f"{cls:<32} | Safe: {str(v['safe']):<5} | {v['status']} | Advice: {v['treatment'][:45]}...")
