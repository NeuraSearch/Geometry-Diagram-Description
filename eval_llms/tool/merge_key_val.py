# coding:utf-8

"""This file provides methods to combine the "diagram_description", "text", "choice_list", etc."""

DIAGRMA_PREFIX =  "Here are the basic description of the diagram: "
CHOICE_PREFIX = "The Choices are: "

def naive_merge(diagram_description: dict, text: dict, choice_list: list) -> str:
    """Naive merge method, just combine all together."""
    to_merge_diagram_description = ""
    to_merge_text = ""
    to_merge_choice = ""
    
    if diagram_description != None:
        # temp = ", ".join(diagram_description["structure"]) + ", ".join(diagram_description["semantic"])
        to_merge_diagram_description = f"{DIAGRMA_PREFIX} {diagram_description}"
        
    if text != None:
        # to_merge_text = text["Num_Exp"]
        to_merge_text = text
    
    if choice_list != None:
        choice_list = [str(choice) for choice in choice_list if choice != None]
        temp = ""
        for i, choice in enumerate(choice_list):
            choice_char = chr(65 + i)
            temp += f"{choice_char}: {choice}, "
        to_merge_choice = f"{CHOICE_PREFIX} {temp}"
    
    return to_merge_diagram_description + "\n" + to_merge_text + "\n" + to_merge_choice