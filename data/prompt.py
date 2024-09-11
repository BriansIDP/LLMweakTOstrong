prompts = {
    "system": "You are a helpful assistant that follows user instruction",
    "task_description": "Consider the following list of slot types provided to you:\n{}\n",
    "delib_query": "Then, consider the following outputs from weak models that provide some possible answers with errors:\n{}\n",
    "query": "Now consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n",
    "systemnbest": "You will be presented with a list of hypotheses from an ASR system for one utterance. Please extract slot and values of that utterance in JSON format.",
    "user2nbest": "Now consider the following candidate hypotheses containing one or more of the above slot types. Can you extract slots belonging to that list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n",
    "format": [
        "{label}.",
        "The answer is {label}.",
        "{label} is the answer",
        "I know that {truepath}, so {label}.",
        "Knowledge: {truepath}. Answer is {label}.",
        "Knowledge I found is {truepath}. Answer: {label}.",
        "I know that {truepath}, so the answer is {label}.",
        "Because {truepath}, {label} is the answer",
        "Based on the knowledge {truepath}, it is {label}.",
    ],
    "direct": "The answer is {label}."
}

templates = {
    "vicuna": {
        "slot": [
            "USER: {taskdesc}{query}\"{content}\"\nASSISTANT:",
            "USER: Consider the following knowledge provided to you as a prior:\n{knowledge}\n{taskdesc}{query}\"{content}\"\nASSISTANT:",
            # "USER: Consider the following knowledge provided to you as a prior:\n{knowledge}\n{taskdesc}For the sentence: \"{content}\", please find the most likely slot and value and output in JSON format.\nASSISTANT:",
            "USER: {taskdesc}For the sentence: \"{content}\", please find the most likely slot and value and output in JSON format.\nASSISTANT:",
        ],
        "qa": [
            "USER: {content}\nASSISTANT:",
            "USER: {content}? You are provided with the knowledge: {truepaths}.\nASSISTANT:",
            "USER: {content}. Please find relevant knowledge.\nASSISTANT:",
        ],
    },
    "llama2": {
        "slot": [
            "[INST] <<SYS>>\n{system}\n<</SYS>>\n{taskdesc}{query}\"{content}\"\n[/INST]\n",
            "[INST] <<SYS>>\n{system}\n<</SYS>>\nConsider the following knowledge provided to you as a prior:\n{knowledge}\n{taskdesc}{query}\"{content}\"\n[/INST]\n",
            "[INST] <<SYS>>\n{system}\n<</SYS>>\nConsider the following knowledge provided to you as a prior:\n{values}\n{taskdesc}{query}\"{content}\"\n[/INST]\n",
        ],
        "qa": [
            "[INST] <<SYS>>\n{system}\n<</SYS>>\n{content}\n[/INST]",
            "[INST] <<SYS>>\n{system}\n<</SYS>>\n{content}\n[/INST]",
            "[INST] <<SYS>>\n{system}\n<</SYS>>\n{content}\n[/INST]",
        ],
    }
}
