## Source: https://github.com/allenai/scispacy/issues/141#issuecomment-518274586
## Author: https://github.com/phosseini
##   File: /mnt/Vancouver/apps/spacy/entity_options.py
##    Env: Python 3.7 venv:
##    Use:
##          import entity_options
##          from entity_options import get_entity_options
##          displacy.serve(doc, style="ent", options=get_entity_options(random_colors=True))
##    Ent: https://github.com/allenai/scispacy/issues/79#issuecomment-557766506 ## CRAFT entities

import random 

def get_entity_options(random_colors=False):
    """ generating color options for visualizing the named entities """

    def color_generator(number_of_colors):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        return color

    entities = ["GGP", "SO", "TAXON", "CHEBI", "GO", "CL", "DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN", \
                "DISEASE", "CHEMICAL", "CANCER", "ORGAN", "TISSUE", "ORGANISM", "CELL", "AMINO_ACID", \
                "GENE_OR_GENE_PRODUCT", "SIMPLE_CHEMICAL", "ANATOMICAL_SYSTEM", "IMMATERIAL_ANATOMICAL_ENTITY", \
                "MULTI-TISSUE_STRUCTURE", "DEVELOPING_ANATOMICAL_STRUCTURE", "ORGANISM_SUBDIVISION", "CELLULAR_COMPONENT"]

    colors = {"ENT":"#E8DAEF"}

    if random_colors:
        color = color_generator(len(entities))
        for i in range(len(entities)):
            colors[entities[i]] = color[i]
    else:
        entities_cat_1 = {"GGP":"#F9E79F", "SO":"#F7DC6F", "TAXON":"#F4D03F", "CHEBI":"#FAD7A0", "GO":"#F8C471", "CL":"#F5B041"}
        entities_cat_2 = {"DNA":"#82E0AA", "CELL_TYPE":"#AED6F1", "CELL_LINE":"#E8DAEF", "RNA":"#82E0AA", "PROTEIN":"#82E0AA"}
        entities_cat_3 = {"DISEASE":"#D7BDE2", "CHEMICAL":"#D2B4DE"}
        entities_cat_4 = {"CANCER":"#ABEBC6", "ORGAN":"#82E0AA", "TISSUE":"#A9DFBF", "ORGANISM":"#A2D9CE", "CELL":"#76D7C4", \
                          "AMINO_ACID":"#85C1E9", "GENE_OR_GENE_PRODUCT":"#AED6F1", "SIMPLE_CHEMICAL":"#76D7C4", "ANATOMICAL_SYSTEM":"#82E0AA", \
                          "IMMATERIAL_ANATOMICAL_ENTITY":"#A2D9CE", "MULTI-TISSUE_STRUCTURE":"#85C1E9", "DEVELOPING_ANATOMICAL_STRUCTURE":"#A9DFBF", \
                          "ORGANISM_SUBDIVISION":"#58D68D", "CELLULAR_COMPONENT":"#7FB3D5"}

        entities_cats = [entities_cat_1, entities_cat_2, entities_cat_3, entities_cat_4]
        for item in entities_cats:
            colors = {**colors, **item}

    options = {"ents": entities, "colors": colors}
    # print(options)
    return options