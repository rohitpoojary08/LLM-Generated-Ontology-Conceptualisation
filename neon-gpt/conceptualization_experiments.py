from api_utils import send_prompt
from neon_gpt_ontology_generation import ONTOLOGY_CONFIGS

def generate_requirements(domain_description, domain_name, persona, keywords, ontology_metrics, model=None):
    prompt = f"""You are a {persona}.

            The ontology domain is:

            DOMAIN NAME:
            {domain_name}

            DOMAIN DESCRIPTION:
            {domain_description}

            KEYWORDS:
            {keywords}

            Use all the keywords, not just a snippet, and your own knowledge
            and domain understanding to generate the ontology requirements
            based on the NeOn methodology.

            The original ontology has:
            {ontology_metrics}

            The NeOn methodology starts by specifying the following ontology
            requirements:

            - The purpose of the ontology
            - The scope of the ontology
            - The target group of the ontology
            - The intended uses
            - The functional requirements
            - The non-functional requirements
        """

    return send_prompt(
        prompt,
        persona,
        "experiment_requirements",
        model=model
    )

def extend_requirements_with_reuse(domain_description, requirements, persona, reuse_example_desc, few_shot_reuse, model=None):
    prompt = f"""
            DOMAIN DESCRIPTION:
            {domain_description}

            ONTOLOGY REQUIREMENTS:
            ###start_requirements###
            {requirements}
            ###end_requirements###

            Based on these ontology specifications developed using the NeOn
            methodology, extend and improve them through ontology reuse principles.

            The ontology should accurately represent the complexity and hierarchical
            structure of the domain. It should be detailed, with well-defined
            hierarchy levels and interconnected relationships between concepts.
            This can be achieved with ontology reuse.
            Reuse refers to utilizing existing ontological knowledge or structures
            as input in the development of new ontologies.
            It allows for more efficient and consistent knowledge representation across applications.
            Reuse the following example to improve the ontology structure from:
            {reuse_example_desc}

            EXAMPLE TRIPLES:
            {few_shot_reuse}
        """

    return send_prompt(
        prompt,
        persona,
        "experiment_requirements_reuse",
        model=model
    )

def generate_cqs(domain_description, requirements, persona, ontology_metrics, few_shot_cqs, model=None):
    prompt = f"""
            DOMAIN DESCRIPTION:
            ###start_domain_description###
            {domain_description}
            ###end_domain_description###

            ONTOLOGY REQUIREMENTS:
            ###start_requirements###
            {requirements}
            ###end_requirements###

            Based on the domain description and ontology requirements above,
            write a list of Competency Questions that the core module of the
            ontology should be able to answer.
            Make the list as complete as possible.
            The original ontology has: {ontology_metrics}
            Make sure that the generated ontology reflects the previous metrics and has a high subclass count;
            if the class count is n, the subclass count should at least be n-1.

            Here are some examples of Competency Questions to guide you: {few_shot_cqs}
        """

    return send_prompt(
        prompt,
        persona,
        "experiment_cqs",
        model=model
    )

def extract_elements(domain_description,cqs,persona,ontology_metrics,few_shot_entity_extraction,model=None):
    prompt = f"""
            DOMAIN DESCRIPTION:
            ###start_domain_description###
            {domain_description}
            ###end_domain_description###

            COMPETENCY QUESTIONS:
            ###start_competency_questions###
            {cqs}
            ###end_competency_questions###

            For each Competency Question, extract the entities, relations (properties), and axioms that must be introduced
            in the ontology.
            A single competency question can help in extracting more than one triple.
            Do this for ALL competency questions, not just a snippet.

            Here are some examples to guide you: {few_shot_entity_extraction}

            The original ontology has: {ontology_metrics}
            Make sure that the generated ontology reflects the previous metrics 
            and has a high subclass count; if the class count is n, 
            the subclass count should at least be n-1.
        """

    return send_prompt(
        prompt,
        persona,
        "experiment_elements",
        model=model
    )

def generate_conceptual_model(domain_description,cqs,elements,persona,ontology_metrics,model=None):
    prompt = f"""
            DOMAIN DESCRIPTION:
            ###start_domain_description###
            {domain_description}
            ###end_domain_description###

            COMPETENCY QUESTIONS:
            ###start_competency_questions###
            {cqs}
            ###end_competency_questions###

            EXTRACTED ENTITIES, RELATIONS, AND AXIOMS:
            ###start_elements###
            {elements}
            ###end_elements###

            Considering all the generated entities, relations (properties),
            and axioms, generate a conceptual model expressing the triples of the entities, properties (relations), 
            and axioms in the format plain-text list of triples.
            Write exactly one triple per line in this format:
            (Subject - relation - Object)

            Do it for all the entities, properties (relations), and axioms — not just a snippet.

            Examples:
            (RedWine - subclassOf - Wine)
            (Wine - hasGrapeVariety - GrapeVariety)
            (hasGrapeVariety - domain - Wine)

            Use names, not RDF prefixes or URIs. Do not output Turtle,
            @prefix declarations, RDF/OWL syntax, Markdown tables, or code fences.
            Put the complete list between ###start_output### and ###end_output###.

            The original ontology has:
            {ontology_metrics}
            Make sure that the generated ontology reflects the previous metrics 
            and has a high subclass count; if the class count is n, 
            the subclass count should at least be n-1.

            Make sure that your conceptual model is logically consistent and free from common pitfalls 
            (e.g., Wrong inverse relationships, Cycles in a class hierarchy, Multiple domains or ranges, 
            and Wrong transitive relationships).
            
            Here are all the extracted triples from the previous step:
        """

    return send_prompt(
        prompt,
        persona,
        "experiment_conceptual_model",
        model=model
    )

def run_experiment(
    domain_description,
    requirements=None,
    cqs=None,
    config=None,
    model=None
):
    if config is None:
        raise ValueError("Ontology configuration must be provided.")

    persona = config["persona"]
    domain_name = config["domain_name"]
    keywords = config["keywords"]
    ontology_metrics = config["ontology_metrics"]

    # ----------------------------------
    # CASE A: Requirements not supplied
    # ----------------------------------
    if requirements is None:

        print("\n=== Generating requirements ===")

        generated_requirements = generate_requirements(
            domain_description=domain_description,
            domain_name=domain_name,
            persona=persona,
            keywords=keywords,
            ontology_metrics=ontology_metrics,
            model=model
        )

        print("\n=== Extending requirements through reuse ===")

        reused_requirements = extend_requirements_with_reuse(
            domain_description=domain_description,
            requirements=generated_requirements,
            persona=persona,
            reuse_example_desc=config["reuse_example_desc"],
            few_shot_reuse=config["few_shot_reuse"],
            model=model
        )

        requirements = (
            generated_requirements
            + "\n\n"
            + reused_requirements
        )

    # ----------------------------------
    # CASE A/B: CQs not supplied
    # ----------------------------------
    if cqs is None:

        print("\n=== Generating competency questions ===")

        cqs = generate_cqs(
            domain_description=domain_description,
            requirements=requirements,
            persona=persona,
            ontology_metrics=ontology_metrics,
            few_shot_cqs=config["few_shot_cqs"],
            model=model
        )

    # ----------------------------------
    # ALWAYS: extract elements
    # ----------------------------------

    print("\n=== Extracting elements ===")

    elements = extract_elements(
        domain_description=domain_description,
        cqs=cqs,
        persona=persona,
        ontology_metrics=ontology_metrics,
        few_shot_entity_extraction=
        config["few_shot_entity_extraction"],
        model=model
    )

    # ----------------------------------
    # ALWAYS: conceptual model
    # ----------------------------------

    print("\n=== Generating conceptual model ===")

    conceptual_model = generate_conceptual_model(
        domain_description=domain_description,
        cqs=cqs,
        elements=elements,
        persona=persona,
        ontology_metrics=ontology_metrics,
        model=model
    )

    return {
        "requirements": requirements,
        "cqs": cqs,
        "elements": elements,
        "conceptual_model": conceptual_model
    }

if __name__ == "__main__":

    config = ONTOLOGY_CONFIGS["Wine"]

    # Case A
    '''result = run_experiment(
        domain_description=config["domain_description"],
        config=config,
        model="mistralai/mistral-large-2512"
    )'''

    # Load requirements generated during Case A
    with open("outputs/Case_A/DeepSeek_Wine_requirements_A.txt","r",encoding="utf-8") as f:
        generated_requirements = f.read()
    
    with open("outputs/Case_A/DeepSeek_Wine_requirements_reuse_A.txt","r",encoding="utf-8") as f:
        reused_requirements = f.read()

    supplied_requirements = (generated_requirements+ "\n\n"+ reused_requirements)

    with open("outputs/Case_A/DeepSeek_Wine_cqs_A.txt","r",encoding="utf-8") as f:
        supplied_cqs = f.read()

    # Case B
    '''result = run_experiment(
        domain_description=config["domain_description"],
        requirements=supplied_requirements,
        config=config, 
        model="deepseek/deepseek-v3.2-exp"
    )'''

    # Case C
    result = run_experiment(
        domain_description=config["domain_description"],
        requirements=supplied_requirements,
        cqs=supplied_cqs,
        config=config,
        model="deepseek/deepseek-v3.2-exp"
    )

    print("\n\n========== FINAL CONCEPTUAL MODEL ==========\n")
    print(result["conceptual_model"])