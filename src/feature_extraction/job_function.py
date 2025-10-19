def extract_job_function_from_title(title: str) -> str:
    """
    Extracts the professional function from a job title by checking for
    specific terms first and falling back to shorter, more ambiguous terms last.
    """
    if not isinstance(title, str):
        return "unknown"

    title_lower = title.lower()

    # --- 1. Check for specific C-suite and executive titles first ---
    if any(
        keyword in title_lower
        for keyword in [
            "chief compliance officer",
            "chief legal officer",
            "chief privacy officer",
            "chief risk officer",
            "chief sustainability officer",
        ]
    ):
        return "legal_risk_compliance"
    if any(
        keyword in title_lower
        for keyword in [
            "chief brand officer",
            "chief communications officer",
            "chief content officer",
            "chief creative officer",
            "chief design officer",
            "chief marketing officer",
            "chief reputation officer",
        ]
    ):
        return "marketing_creative"
    if any(
        keyword in title_lower
        for keyword in [
            "chief administrative officer",
            "chief operating officer",
            "chief process officer",
            "chief restructuring officer",
            "chief services officer",
            "chief visibility officer",
        ]
    ):
        return "operations"
    if any(
        keyword in title_lower
        for keyword in [
            "chief customer officer",
            "chief experience officer",
            "chief innovation officer",
            "chief product officer",
        ]
    ):
        return "product_experience"
    if any(
        keyword in title_lower
        for keyword in [
            "chief business development officer",
            "chief commercial officer",
            "chief growth officer",
            "chief revenue officer",
        ]
    ):
        return "sales_business_development"
    if any(
        keyword in title_lower
        for keyword in [
            "chief analytics officer",
            "chief data officer",
            "chief genealogical officer",
            "chief research officer",
            "chief scientific officer",
        ]
    ):
        return "science_data"
    if any(keyword in title_lower for keyword in ["chief security officer"]):
        return "security"
    if any(keyword in title_lower for keyword in ["chief supply chain officer"]):
        return "supply_chain"
    if any(keyword in title_lower for keyword in ["chief quality officer"]):
        return "quality_assurance"
    if any(
        keyword in title_lower
        for keyword in [
            "chief confluence officer",
            "chief digital officer",
            "chief information officer",
            "chief information security officer",
            "chief solutions officer",
            "chief technical officer",
            "chief technology officer",
            "chief technology security officer",
            "chief web officer",
        ]
    ):
        return "engineering"
    if any(
        keyword in title_lower
        for keyword in ["chief financial officer", "chief investment officer"]
    ):
        return "finance_accounting"
    if any(
        keyword in title_lower
        for keyword in [
            "executive",
            "partner",
            "principal",
            "chief business officer",
            "chief executive officer",
            "chief strategy officer",
            "chief visionary officer",
            "chief innovation officer",
            "chief product officer",
        ]
    ):
        return "general_management"
    if any(
        keyword in title_lower
        for keyword in [
            "chief diversity officer",
            "chief human resources officer",
            "chief learning officer",
            "chief people officer",
        ]
    ):
        return "human_resources_hr"

    # --- 2. Fallback to general, more specific keywords ---
    if any(keyword in title_lower for keyword in ["data scientist", "data science", "predictive modeler"]):
        return "data_scientist"

    # Check for higher-prestige legal roles first
    if any(keyword in title_lower for keyword in ['attorney', 'lawyer', 'counsel', 'litigation', 'negotiator']):
        return "legal_attorney_counsel"
    # Fallback to legal support and general legal terms
    if any(keyword in title_lower for keyword in ['legal', 'paralegal', 'reviewer', 'court reporter']):
        return "legal_support"

    if any(
        keyword in title_lower
        for keyword in ["scientist", "research", "chemist", "biologist", "ecologist", "geophysicist", "mathematician", "lab", "laboratory", "math", "fish"]
    ):
        return "science_research"
    if any(
        keyword in title_lower
        for keyword in ["quality assurance", "quality control", "tester", "auditor"]
    ):
        return "quality_assurance"
    if any(
        keyword in title_lower
        for keyword in [
            "administrator", "architect", "code", "coding", "cyber security", "data modeler",
            "developer", "engineer", "information technology", "programmer", "scrum master",
            "sdet", "sre", "webmaster", "wordpress", "hadoop", "jira", "netbackup", "sap",
            "sharepoint", "ucce", "workday", "data validator", "frontend", "front end", "database", "nuclear"
        ]
    ):
        return "engineering_it"

    # --- New & Expanded Healthcare Sections ---
    if any(keyword in title_lower for keyword in ['surgeon', 'cardiologist', 'dermatologist', 'neurologist', 'oncology', 'radiologist', 'anesthesiologist', 'pathologist', 'medical director', 'ob/gyn', 'obgyn', 'surgery', 'pediatric', 'cardiovascular', 'neuroscience', 'neurosurgery', 'endoscopy', 'endodontist', 'radiology', 'vascular', 'urology', 'physiatrist']):
        return "healthcare_specialist_physician"
    if any(keyword in title_lower for keyword in ['physician', 'doctor', 'veterinarian', 'psychiatrist', 'dentist', 'orthodontist', 'resident']):
        return "healthcare_general_physician"
    if any(keyword in title_lower for keyword in ['pharmacist', 'optometrist', 'psychologist', 'therapist', 'dietitian', 'chiropractor', 'clinician', 'nurse practitioner', 'physician assistant', 'audiologist', 'pathologist', 'psychometrician', 'therapy', 'wellness', 'audiology']):
        return "healthcare_advanced_practice"
    if any(keyword in title_lower for keyword in ['nurse', 'nursing', 'technologist', 'sonographer', 'paramedic', 'emt', 'technician', 'dental hygienist', 'hygienist', 'radiologic', 'surgical tech', 'nutritionist', 'echocardiographer', 'mammography', 'polysomnographer', 'palliative']):
        return "healthcare_nursing_allied"
    if any(keyword in title_lower for keyword in ['phlebotomist', 'caregiver', 'nanny', 'provider', 'aide', 'medical assistant', 'patient care', 'home health', 'personal care', 'phlebotomy', 'care']):
        return "healthcare_support"
    if any(keyword in title_lower for keyword in ['healthcare', 'medical', 'clinical', 'patient', 'pharmacy', 'surgical', 'dental', 'ambulatory', 'telemedicine', 'clinic']):
        return "healthcare_other"

    if any(
        keyword in title_lower
        for keyword in [
            "finance", "financial", "investment", "accounting", "accountant", "investor", "tax",
            "auditor", "banker", "teller", "reinsurance", "controller", "payroll", "bookkeeper",
            "billing", "adjuster", "appraiser", "actuary", "advisor", "loan", "mortgage", "collections",
            "trader", "derivatives", "fixed income", "treasury", "actuarial", "valuations", "chargeback",
            "broker", "economist"
        ]
    ):
        return "finance_accounting"
    if any(keyword in title_lower for keyword in ["insurance", "claims"]):
        return "insurance"
    if any(
        keyword in title_lower
        for keyword in ["compliance", "regulatory", "credentialing", "kyc"]
    ):
        return "compliance_regulatory"
    if any(keyword in title_lower for keyword in ["environmental health", "safety", "hazardous materials"]):
        return "safety_environmental"
    if any(
        keyword in title_lower
        for keyword in [
            "human resources", "talent", "recruiter", "employee", "people operations",
            "onboarding", "training", "benefits", "generalist"
        ]
    ):
        return "hr"
    if any(
        keyword in title_lower
        for keyword in [
            "marketing", "creative", "content", "writer", "designer", "communications", "social media",
            "editor", "producer", "art director", "brand ambassador", "stylist", "strategist", "seo",
            "paid search", "proofreader", "news", "reporter"
        ]
    ):
        return "marketing_creative"
    if any(
        keyword in title_lower
        for keyword in ["sales", "account", "business development", "setter", "acct. exec", "agent", "business"]
    ):
        return "sales"
    if any(
        keyword in title_lower
        for keyword in [
            "supply chain", "logistics", "warehouse", "sourcing", "shipper", "receiving",
            "buyer", "procurement", "inventory", "dispatcher", "selector", "filler", "purchasing",
            "merchandiser", "delivery", "planner", "forwarder", "freight", "vendor", "shipping"
        ]
    ):
        return "supply_chain"
    if any(
        keyword in title_lower
        for keyword in [
            "technician", "estimator", "welder", "driver", "handler", "maintenance", "mechanic",
            "inspector", "assembler", "electrician", "operator", "coiling", "custodian",
            "janitor", "machinist", "laborer", "plumber", "carpenter", "installer", "locksmith",
            "painter", "fabricator", "detailer", "cleaner", "splicer", "groundskeeper", "caretaker",
            "landscaper", "manufacturing", "millwright", "worker", "toolmaker", "tool & die", "hvac",
            "rigger", "roofer", "jeweler", "truck", "meat", "forklift", "gardener"
        ]
    ):
        return "skilled_trades"
    if any(
        keyword in title_lower
        for keyword in [
            "housekeeper", "attendant", "hospitality", "busser", "server", "house person", "aide",
            "cook", "chef", "dishwasher", "barista", "bartender", "host", "valet", "groomer",
            "trainer", "food service", "crewmember", "lifeguard", "baker", "esthetician", "fryer",
            "bakery", "deli", "beauty", "concierge", "culinary", "housekeeping", "waxing"
        ]
    ):
        return "service_hospitality"
    if any(
        keyword in title_lower
        for keyword in [
            "administrative", "assistant", "customer service", "support", "representative",
            "coordinator", "clerk", "examiner", "office manager", "receptionist", "data entry",
            "front desk", "scheduler", "clerical", "client service", "customer success",
            "help desk", "contact center", "documentation", "records", "advocate", "liaison",
            "secretary", "mail", "mailroom", "desk"
        ]
    ):
        return "admin_support"
    if any(
        keyword in title_lower
        for keyword in [
            "security", "protection", "investigator", "police", "correctional",
            "officer", "assessor", "fedramp"
        ]
    ):
        return "security"
    if any(keyword in title_lower for keyword in ["real estate", "leasing", "property"]):
        return "real_estate"
    if any(
        keyword in title_lower
        for keyword in [
            "store", "retail", "merchant", "cashier", "team member", "stock",
            "keyholder", "checker"
        ]
    ):
        return "retail"
    if any(
        keyword in title_lower
        for keyword in [
            "faculty", "instructor", "teacher", "proctor", "educator", "tutor",
            "coach", "dean", "paraprofessional", "mentor"
        ]
    ):
        return "education"
    if any(
        keyword in title_lower
        for keyword in ["social work", "case manager", "counselor", "behavior", "youth", "chaplain"]
    ):
        return "social_services"

    # --- New Categories Based on Extracted Keywords ---
    if any(keyword in title_lower for keyword in ['pilot', 'aviation', 'flight']):
        return "aviation"
    if any(keyword in title_lower for keyword in ['linguist', 'translator', 'interpreter']):
        return "linguistics_translation"
    if any(keyword in title_lower for keyword in ['photographer', 'animator', 'artist', 'illustrator', 'retoucher']):
        return "creative_arts"
    if any(keyword in title_lower for keyword in ['curator', 'archivist', 'librarian']):
        return "archive_curation"
    if any(keyword in title_lower for keyword in ['surveyor', 'landman']):
        return "surveying_land"

    if any(
        keyword in title_lower
        for keyword in [
            "manager", "management", "director", "supervisor", "lead", "vp",
            "vice president", "executive", "chief", "superintendent", "foreman",
            "head", "partner", "principal"
        ]
    ):
        return "management_leadership"

    # --- 3. Fallback for single, less specific keywords ---
    if "analyst" in title_lower or "analytics" in title_lower:
        return "analyst"
    if "product" in title_lower:
        return "product"
    if "consultant" in title_lower:
        return "consulting"
    if "operations" in title_lower:
        return "operations"
    if "project" in title_lower:
        return "project_management"
    if "strategy" in title_lower:
        return "strategy"

    # --- 4. Final fallback for very short, ambiguous keywords ---
    if any(keyword in title_lower for keyword in ["qa", "qc"]):
        return "quality_assurance"
    if any(keyword in title_lower for keyword in ["it", "dev", "dba", "gis", "cad"]):
        return "engineering_it"
    if "hr" in title_lower:
        return "hr"
    if "ds" in title_lower:
        return "data_scientist"
    if any(keyword in title_lower for keyword in ["rn", "lpn", "cna", "mri", "health", "pt", "ot", "lvn", "cma", "pta", "cota", "lmsw", "licsw", "lmft", "lmhc"]):
        return "healthcare"
    if any(keyword in title_lower for keyword in ["cfo", "cpa"]):
        return "finance_accounting"
    if "ehs" in title_lower:
        return "safety_environmental"
    if "bio" in title_lower:
        return "biology"
    if any(keyword in title_lower for keyword in ["ceo", "cso", "cvo", "cbo"]):
        return "general_management"

    if any(keyword in title_lower for keyword in ["specialist", "specialists"]):
        return "specialist"

    if "associate" in title_lower:
        return "associate"

    return "other"