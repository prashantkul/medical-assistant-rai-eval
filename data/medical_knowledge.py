"""
Medical Knowledge Base
Sample medical documents for the RAG system.
These are simplified educational examples - NOT for actual medical use.
"""

MEDICAL_DOCUMENTS = [
    {
        "id": "diabetes_type2_overview",
        "title": "Type 2 Diabetes Overview",
        "content": """
        Type 2 diabetes is a chronic condition that affects how your body metabolizes sugar (glucose).
        With type 2 diabetes, your body either resists the effects of insulin or doesn't produce
        enough insulin to maintain normal glucose levels.

        Risk factors include:
        - Being overweight or obese
        - Family history of diabetes
        - Age 45 or older
        - Physical inactivity
        - High blood pressure

        Management includes:
        - Healthy eating and meal planning
        - Regular physical activity (150 minutes per week)
        - Blood sugar monitoring
        - Diabetes medications or insulin therapy when needed
        - Regular check-ups with healthcare provider

        Complications if unmanaged can include heart disease, nerve damage, kidney disease,
        and vision problems.
        """,
        "category": "chronic_disease"
    },
    {
        "id": "hypertension_guidelines",
        "title": "Hypertension Management Guidelines",
        "content": """
        Hypertension (high blood pressure) is defined as blood pressure readings consistently
        at or above 130/80 mmHg.

        Blood Pressure Categories:
        - Normal: Less than 120/80 mmHg
        - Elevated: 120-129/<80 mmHg
        - Stage 1 Hypertension: 130-139/80-89 mmHg
        - Stage 2 Hypertension: ≥140/90 mmHg
        - Hypertensive Crisis: >180/120 mmHg (seek immediate care)

        Lifestyle modifications:
        - Reduce sodium intake (less than 2,300 mg per day)
        - DASH diet (Dietary Approaches to Stop Hypertension)
        - Regular aerobic exercise
        - Limit alcohol consumption
        - Maintain healthy weight
        - Stress management

        Medications may include ACE inhibitors, ARBs, calcium channel blockers, or diuretics
        as prescribed by a healthcare provider.
        """,
        "category": "cardiovascular"
    },
    {
        "id": "common_cold_care",
        "title": "Common Cold Treatment and Care",
        "content": """
        The common cold is a viral infection of the upper respiratory tract. Most people recover
        within 7-10 days without medical treatment.

        Symptoms include:
        - Runny or stuffy nose
        - Sore throat
        - Cough
        - Mild headache
        - Sneezing
        - Low-grade fever (more common in children)
        - Fatigue

        Self-care measures:
        - Rest and get adequate sleep
        - Stay hydrated with water, warm liquids, and broths
        - Gargle with salt water for sore throat
        - Use humidifier for congestion
        - Over-the-counter pain relievers (acetaminophen, ibuprofen) for aches and fever

        When to see a doctor:
        - Symptoms lasting more than 10 days
        - Severe or unusual symptoms
        - High fever (>101.3°F or 38.5°C)
        - Difficulty breathing
        - Severe sore throat or headache

        Note: Antibiotics do NOT work for common colds as they are caused by viruses.
        """,
        "category": "infectious_disease"
    },
    {
        "id": "medication_safety",
        "title": "Medication Safety Guidelines",
        "content": """
        Proper medication use is critical for treatment effectiveness and safety.

        Key principles:
        - Take medications exactly as prescribed by your healthcare provider
        - Never share prescription medications with others
        - Do not stop medications without consulting your doctor
        - Keep a current list of all medications, including over-the-counter drugs and supplements
        - Store medications properly (temperature, light exposure)
        - Check expiration dates regularly

        Drug interactions:
        - Some medications can interact with food, alcohol, or other drugs
        - Always inform healthcare providers of all medications you take
        - Use the same pharmacy when possible for interaction checking

        Side effects:
        - Report any unusual symptoms to your healthcare provider
        - Know the difference between common side effects and serious adverse reactions
        - Keep emergency contact information accessible

        Disposal:
        - Do not flush medications down the toilet unless specifically instructed
        - Use medication take-back programs when available
        - Mix unused medications with undesirable substances (coffee grounds, cat litter) before disposal
        """,
        "category": "medication_safety"
    },
    {
        "id": "mental_health_depression",
        "title": "Depression: Understanding and Treatment",
        "content": """
        Depression is a common but serious mood disorder that affects how you feel, think, and
        handle daily activities.

        Symptoms (must persist for at least 2 weeks):
        - Persistent sad, anxious, or "empty" mood
        - Loss of interest in activities once enjoyed
        - Changes in appetite or weight
        - Sleep disturbances (insomnia or oversleeping)
        - Fatigue or loss of energy
        - Feelings of worthlessness or guilt
        - Difficulty concentrating or making decisions
        - Thoughts of death or suicide

        Treatment options:
        - Psychotherapy (cognitive behavioral therapy, interpersonal therapy)
        - Medications (antidepressants prescribed by a psychiatrist or doctor)
        - Combination of therapy and medication (often most effective)
        - Lifestyle changes (exercise, sleep hygiene, stress reduction)
        - Support groups

        Emergency: If you or someone you know is having thoughts of suicide, call the
        National Suicide Prevention Lifeline at 988 immediately.

        Recovery is possible with proper treatment and support.
        """,
        "category": "mental_health"
    },
    {
        "id": "preventive_care",
        "title": "Preventive Healthcare Screenings",
        "content": """
        Preventive care helps detect health problems early when treatment is most effective.

        Recommended screenings (general guidelines):

        Adults 18-39:
        - Blood pressure: Every 2 years (if normal)
        - Cholesterol: Every 5 years starting at age 20
        - Diabetes: Every 3 years starting at age 35
        - Dental checkups: Annually
        - Eye exams: As needed or every 2 years

        Adults 40-64:
        - All of the above, plus:
        - Mammogram (women): Every 1-2 years starting at 40
        - Colonoscopy: Starting at age 45, then every 10 years
        - Bone density (women): Discuss with provider based on risk factors

        Adults 65+:
        - All of the above, plus:
        - Bone density: Every 2 years
        - Hearing test: Annually
        - Falls risk assessment

        Vaccinations:
        - Flu vaccine: Annually
        - COVID-19: As recommended
        - Tdap/Td: Every 10 years
        - Shingles: Age 50+
        - Pneumonia: Age 65+

        Note: Individual recommendations may vary based on personal and family health history.
        Always consult your healthcare provider for personalized screening schedules.
        """,
        "category": "preventive_care"
    },
    {
        "id": "exercise_guidelines",
        "title": "Physical Activity Guidelines for Adults",
        "content": """
        Regular physical activity is one of the most important things you can do for your health.

        Recommended amounts for adults:
        - Aerobic activity: At least 150 minutes of moderate-intensity OR 75 minutes of
          vigorous-intensity per week
        - Muscle-strengthening: Activities on 2 or more days per week

        Types of activities:
        Moderate intensity:
        - Brisk walking (2.5-4 mph)
        - Water aerobics
        - Recreational swimming
        - Cycling on level terrain
        - Gardening

        Vigorous intensity:
        - Jogging or running
        - Swimming laps
        - Cycling fast or uphill
        - Singles tennis
        - Aerobic dancing

        Benefits include:
        - Reduced risk of heart disease, stroke, type 2 diabetes
        - Better weight management
        - Stronger bones and muscles
        - Improved mental health and mood
        - Increased longevity

        Getting started:
        - Start slowly if currently inactive
        - Choose activities you enjoy
        - Set realistic goals
        - Track your progress
        - Consult your doctor before starting a new exercise program, especially if you have
          chronic conditions or concerns
        """,
        "category": "lifestyle"
    },
]


def get_all_documents():
    """Return all medical documents."""
    return MEDICAL_DOCUMENTS


def get_documents_by_category(category):
    """Get documents filtered by category."""
    return [doc for doc in MEDICAL_DOCUMENTS if doc["category"] == category]


def get_document_by_id(doc_id):
    """Get a specific document by ID."""
    for doc in MEDICAL_DOCUMENTS:
        if doc["id"] == doc_id:
            return doc
    return None
