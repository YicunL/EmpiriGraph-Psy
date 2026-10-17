"""Offline, dependency-aware v10 leave-one-stage-out prompt builders.

No API client is loaded. A caller can supply a recorded/checkpointed model-call
callback to execute a pipeline later. Prompts come from the v10
notebook; only function definitions are loaded, never notebook setup or secrets.
"""
from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'notebooks/extraction_pipeline_v10.ipynb'
STEPS = {
    1: 'step1_detect_variables',
    2: 'step2_hierarchy_and_normalize',
    3: 'step3_extract_relevant_sentences',
    4: 'step4_extract_relationships',
    5: 'step5_1_validate_relationships',
}
VARIANTS = {'full_12345': (1, 2, 3, 4, 5),
            'omit1_2345': (2, 3, 4, 5),
            'omit2_1345': (1, 3, 4, 5),
            'omit3_1245': (1, 2, 4, 5),
            'omit4_1235': (1, 2, 3, 5),
            'omit5_1234': (1, 2, 3, 4)}


def _replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'Frozen prompt anchor changed: {old[:70]!r}')
    return text.replace(old, new, 1)


def _replace_block(text: str, tag: str, replacement: str = '') -> str:
    result, n = re.subn(rf'<{tag}>.*?</{tag}>', lambda _: replacement,
                        text, flags=re.DOTALL)
    if n != 1:
        raise ValueError(f'Expected one {tag} block; got {n}')
    return result


def _section(text: str, tag: str) -> str:
    found = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.DOTALL)
    if not found:
        raise ValueError(f'Missing {tag} block')
    return found.group(1).strip()


class V10AblationPrompts:
    def __init__(self, source: Path = SOURCE):
        self.source = Path(source)
        self.ns = {'json': json, 'dedent': dedent, 'Tuple': Tuple,
                   'Dict': Dict, 'List': List, 're': re,
                   'extract_token_usage': lambda _: {},
                   'parse_json_response': lambda _: {}}
        notebook = json.loads(self.source.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell.get('cell_type') != 'code':
                continue
            text = ''.join(cell['source'])
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in tree.body:
                wanted_function = isinstance(node, ast.FunctionDef) and node.name in [*STEPS.values(), 'clean_abstract']
                wanted_regex = (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id in {'_CLEAN_RE', '_IMPACT_RE'} for t in node.targets))
                if wanted_function or wanted_regex:
                    module = ast.Module(body=[node], type_ignores=[])
                    exec(compile(module, str(self.source), 'exec'), self.ns)
        if any(name not in self.ns for name in STEPS.values()):
            raise ValueError('Not all five v10 stage functions found')

    def clean_abstract(self, raw: str) -> str:
        return self.ns['clean_abstract'](raw)

    def capture(self, stage: int, *args, **kwargs) -> list[dict]:
        captured = []

        def fake(prompt_blocks, **_):
            captured.extend(copy.deepcopy(prompt_blocks))
            return {'content': '{}', 'usage': None}

        self.ns['call_openai_with_json_complex'] = fake
        self.ns['call_openai_with_json_simple'] = fake
        self.ns[STEPS[stage]](*args, **kwargs)
        return captured

    @staticmethod
    def variables(state: dict[int, dict]) -> list[str]:
        if 2 not in state and 4 in state and 'final_variable_list' in state[4]:
            return state[4]['final_variable_list']
        return state.get(2, state.get(1, {})).get('final_variable_list', [])

    @staticmethod
    def hierarchy(state: dict[int, dict]) -> list[list[str]]:
        if 2 not in state:
            return copy.deepcopy(state.get(4, {}).get('hierarchy', []))
        rels = state.get(2, {}).get('hierarchy', {}).get('hierarchical_relationships', [])
        return [[h['higher_level_variable'], 'hierarchy', h['lower_level_variable']]
                for h in rels]

    def build(self, variant: str, stage: int, abstract: str,
              state: dict[int, dict]) -> list[dict]:
        active = VARIANTS[variant]
        if stage not in active:
            raise ValueError(f'Stage {stage} is omitted in {variant}')
        expected = set(active[:active.index(stage)])
        if set(state) != expected:
            raise ValueError(f'Stage {stage} requires exactly {sorted(expected)}; got {sorted(state)}')
        variables = self.variables(state)
        sentences = state.get(3, {}).get('relevant_sentences', [])
        if stage == 1:
            return self.capture(1, abstract)
        if stage == 2:
            blocks = self.capture(2, abstract, state.get(1, {}))
            if variant == 'omit1_2345':
                # Replace the dependency, not the hierarchy definitions or schema.
                d, u = blocks[0]['content'], blocks[1]['content']
                d = _replace(d, 'Workflow:',
                    'Workflow:\n'
                    '0. No preliminary variable list is supplied. Identify the study variables '
                    'directly from this abstract and assign their IV, DV, moderator or mediator '
                    'roles where the text supports these roles. Include only constructs '
                    'explicitly investigated, measured, categorized, manipulated or substantively '
                    'reported. Exclude incidental, contextual and generic narrative phrases. '
                    'Use this initial inventory for the remaining normalization and hierarchy rules.')
                d = _replace(d,
                    'Do NOT change a variable\'s role position. If a variable is an Independent variable in the input variable list, do not move it to moderation or other positions.',
                    'Assign each variable\'s role from the abstract; there is no earlier role assignment to preserve.')
                u = _replace(u, 'You are given an abstract and a variable list.',
                             'You are given only an abstract. Establish its variable inventory within this call.')
                u = _replace_block(u, 'input_variables')
                blocks[0]['content'], blocks[1]['content'] = d, u
            return blocks
        if stage == 3:
            return self.capture(3, abstract, variables)
        if stage == 4:
            blocks = self.capture(4, abstract, variables, sentences)
            if variant == 'omit2_1345':
                normalization = self.capture(2, abstract, state[1])
                definitions = _section(normalization[0]['content'], 'definitions')
                level_rules = _section(normalization[0]['content'], 'rules').split('Workflow:', 1)[0]
                d, u = blocks[0]['content'], blocks[1]['content']
                d = _replace(d, '- Only extract relationships where ALL variables are in the VARIABLE LIST.',
                    '- Begin with the supplied raw VARIABLE LIST. Also extract hierarchy '
                    'relationships in this call, using the hierarchy rules below and the full '
                    'abstract. Add an explicitly mentioned higher/lower-level variable only '
                    'when it is an endpoint of a valid hierarchy. Include these endpoints '
                    'in final_variable_list. Use that resulting inventory for ALL relationships. '
                    'Do not otherwise split, rename or normalize the supplied variable inventory.')
                d += '\n<hierarchy_rules>\n' + definitions + '\n' + level_rules + '\n</hierarchy_rules>\n'
                u = _replace(u, 'Extract all directional, correlational, and moderation relationships from the abstract.',
                    'Extract hierarchy together with all directional, correlational and moderation relationships from the abstract.')
                u = _replace(u, 'Step 0: Keep only variables from the VARIABLE LIST.',
                    'Step 0: Start with the raw VARIABLE LIST. Find valid hierarchy edges '
                    'using the full abstract and add only their explicitly mentioned endpoints '
                    'to final_variable_list. Keep the original names of all supplied variables.')
                u = _replace(u, 'Step 6: Remove any relationship using variables outside the list.',
                    'Step 6: Remove any relationship using variables outside final_variable_list.')
                u = _replace(u, '{\n  "directional": [',
                    '{\n  "hierarchy": [["upper_var", "hierarchy", "lower_var"]],\n'
                    '  "final_variable_list": ["var1", "var2"],\n  "directional": [')
                d = d.replace('- Use exact variable names from VARIABLE LIST',
                              '- Use exact variable names from final_variable_list')
                blocks[0]['content'], blocks[1]['content'] = d, u
            elif variant == 'omit3_1245':
                d, u = blocks[0]['content'], blocks[1]['content']
                d = _replace(d,
                    'USE RELEVANT SENTENCES\n- Ground all relationships primarily in the provided relevant sentences.',
                    'USE THE ABSTRACT DIRECTLY\n'
                    '- No sentence list is supplied. Ground relationships directly '
                    'in this study\'s hypotheses, methods, analyses and findings in the full abstract. '
                    'Exclude prior-work, background and implication-only claims; retain sentences '
                    'that also contain this study\'s empirical content. Do not output an evidence-sentence list.')
                u = _replace(u, 'Step 1: Use RELEVANT SENTENCES as primary evidence.',
                             'Step 1: Use the full ABSTRACT directly as evidence for this study\'s empirical relationships.')
                u = _replace_block(u, 'evidence')
                blocks[0]['content'], blocks[1]['content'] = d, u
            return blocks
        # Step 5: use the display strings from the batch notebook.
        current = state.get(4, {})
        direction, correlation, moderation = relationship_strings(current)
        h = '; '.join(f'{r[0]} -> {r[2]}' for r in self.hierarchy(state))
        blocks = self.capture(5, abstract=abstract, step2_hierarchy=h,
                              step4_directional=direction, step4_correlational=correlation,
                              step4_moderation=moderation, variable_list='; '.join(variables))
        if variant == 'omit4_1235':
            # A reviewer with empty candidates is not a meaningful four-call pipeline.
            # Merge candidate extraction into step 5 and disclose this changed role.
            extraction = self.capture(4, abstract, variables, sentences)
            initial = ('\n<initial_extraction>\nThere is no earlier relationship draft. '
                       'Within this single call, first extract candidate relationships using '
                       'the rules below, then apply the validation rules to your candidates. '
                       'Return only the final graph in the existing output schema.\n'
                       + _section(extraction[0]['content'], 'rules')
                       + '\n</initial_extraction>\n')
            blocks[0]['content'] = initial + blocks[0]['content']
            blocks[1]['content'] = _replace_block(blocks[1]['content'], 'task',
                '<task>Extract relationships directly from the abstract and provided empirical '
                'sentences, then validate your own candidates within this call.</task>')
            blocks[1]['content'] = _replace_block(blocks[1]['content'], 'current_state',
                '<current_hierarchy>\n' + h + '\n</current_hierarchy>\n'
                '<evidence>\nRELEVANT SENTENCES:\n' + json.dumps(sentences, ensure_ascii=False)
                + '\n</evidence>')
            blocks[1]['content'] = _replace(blocks[1]['content'],
                'Check whether the previous coding incorrectly treated',
                'Check whether your initial candidate coding incorrectly treated')
        return blocks


def relationship_strings(graph: dict) -> tuple[str, str, str]:
    directional = '; '.join(f'{r[0]} -> {r[2]} [{r[3]}]' for r in graph.get('directional', []))
    correlational = '; '.join(f'{r[0]} <-> {r[2]} [{r[3]}]' for r in graph.get('correlational', []))
    moderation = '; '.join(f"{m['moderator']} moderates ({', '.join(m['moderated_variables'])}) [{m['validation']}]"
                           for m in graph.get('moderation', []))
    return directional, correlational, moderation


def validate_stage_result(stage: int, result: dict) -> None:
    """Reject missing fields rather than disguising failures as valid empty graphs."""
    fields = {
        1: ['independent_variables', 'dependent_variables', 'moderators', 'mediators', 'final_variable_list'],
        2: ['hierarchy', 'final_variable_list', 'independent_variables', 'dependent_variables',
            'moderators', 'mediators', 'other_variables_controls'],
        3: ['relevant_sentences'],
        4: ['directional', 'correlational', 'moderation'],
        5: ['hierarchy', 'directional', 'correlational', 'moderation', 'justifications'],
    }[stage]
    if not isinstance(result, dict) or any(k not in result for k in fields):
        raise ValueError(f'Stage {stage} is missing required output fields')
    for k in fields:
        if stage == 2 and k == 'hierarchy':
            if not isinstance(result[k], dict) or not isinstance(result[k].get('hierarchical_relationships'), list):
                raise ValueError('Stage 2 hierarchy must contain hierarchical_relationships')
            for edge in result[k]['hierarchical_relationships']:
                if not isinstance(edge, dict) or any(not isinstance(edge.get(n), str) or not edge[n].strip()
                        for n in ['higher_level_variable', 'lower_level_variable']):
                    raise ValueError('Malformed stage 2 hierarchy edge')
        elif not isinstance(result[k], list):
            raise ValueError(f'Stage {stage} field {k} must be a list')
    string_fields = (fields if stage in (1, 3) else
                     ['final_variable_list', 'moderators', 'mediators'] if stage == 2 else [])
    for k in string_fields:
        if any(not isinstance(v, str) or not v.strip() for v in result[k]):
            raise ValueError(f'Stage {stage} field {k} has an invalid string')
    if stage in (4, 5):
        # The archived v10 outputs also use arrows and verbal relation markers.
        # Category keys determine edge type; keep these markers unchanged.
        for k in ['directional', 'correlational']:
            for edge in result[k]:
                if (not isinstance(edge, list) or len(edge) != 4
                    or any(not isinstance(x, str) or not x.strip() for x in edge)
                    or edge[3] not in {'validated', 'null', 'hypothesized'}):
                    raise ValueError(f'Malformed stage {stage} {k} edge')
        for edge in result['moderation']:
            if (not isinstance(edge, dict) or not isinstance(edge.get('moderator'), str)
                or not edge['moderator'].strip() or not isinstance(edge.get('moderated_variables'), list)
                or len(edge['moderated_variables']) < 2
                or any(not isinstance(x, str) or not x.strip() for x in edge['moderated_variables'])
                or edge.get('validation') not in {'validated', 'null', 'hypothesized'}):
                raise ValueError(f'Malformed stage {stage} moderation edge')
    if stage == 5 or (stage == 4 and 'hierarchy' in result):
        if not isinstance(result['hierarchy'], list):
            raise ValueError(f'Stage {stage} hierarchy must be a list')
        for edge in result['hierarchy']:
            if (not isinstance(edge, list) or len(edge) != 3 or edge[1] != 'hierarchy'
                or any(not isinstance(x, str) or not x.strip() for x in edge)):
                raise ValueError('Malformed stage 5 hierarchy edge')
    if stage == 5:
        if any(not isinstance(j, dict) or not isinstance(j.get('change'), str)
               or not isinstance(j.get('reason'), str) for j in result['justifications']):
            raise ValueError('Malformed stage 5 justification')


def final_graph(variant: str, state: dict[int, dict]) -> dict:
    if set(state) != set(VARIANTS[variant]):
        raise ValueError('All retained stages must finish before export')
    if 5 in state:
        graph = copy.deepcopy(state[5])
        return graph
    # Omit 5: schema conversion only. No filtering, merging, retyping or extra call.
    return {'hierarchy': V10AblationPrompts.hierarchy(state),
            **{k: copy.deepcopy(state[4][k]) for k in ['directional', 'correlational', 'moderation']},
            'justifications': []}


def evaluator_row(variant: str, state: dict[int, dict], article_id: int,
                  title: str, abstract: str) -> dict:
    graph = final_graph(variant, state)
    d, c, m = relationship_strings(graph)
    return {'article_id': f'json_{article_id}', 'title': title, 'abstract': abstract,
            'ablation_variant': variant, 'retained_steps': ''.join(map(str, VARIANTS[variant])),
            'step2_final_vars': '; '.join(V10AblationPrompts.variables(state)),
            'final_variable_list_json': json.dumps(V10AblationPrompts.variables(state), ensure_ascii=False),
            'step5_1_hierarchy': '; '.join(f'{h[0]} -> {h[2]}' for h in graph.get('hierarchy', [])),
            'step5_1_directional': d, 'step5_1_correlational': c, 'step5_1_moderation': m,
            'step5_1_justifications': ' | '.join(f"{j.get('change', '')}: {j.get('reason', '')}"
                                              for j in graph.get('justifications', []))}


def run_pipeline(variant: str, abstract: str,
                 complete: Callable[[int, list[dict]], dict],
                 prefix: dict[int, dict] | None = None,
                 builder: V10AblationPrompts | None = None) -> tuple[dict, dict]:
    """Model-independent orchestration; prefix must be a contiguous retained prefix.

    The supplied callback owns API settings, raw responses, usage, checkpoints and
    retries. This function itself cannot access an API or human annotations.
    Prefix reuse is allowed only with the same inputs/configuration/prompt hashes;
    the caller must check provenance before supplying recorded prefix outputs.
    """
    active = VARIANTS[variant]
    state = copy.deepcopy(prefix or {})
    if set(state) != set(active[:len(state)]):
        raise ValueError('Reused outputs must form a contiguous retained prefix')
    for stage, result in state.items():
        validate_stage_result(stage, result)
    prompts = builder or V10AblationPrompts()
    for stage in active[len(state):]:
        blocks = prompts.build(variant, stage, abstract, state)
        result = complete(stage, blocks)
        validate_stage_result(stage, result)
        if variant == 'omit2_1345' and stage == 4:
            if not isinstance(result.get('hierarchy'), list) or not isinstance(result.get('final_variable_list'), list):
                raise ValueError('1345 step 4 must return hierarchy and final_variable_list')
            if any(not isinstance(v, str) or not v.strip() for v in result['final_variable_list']):
                raise ValueError('1345 step 4 has invalid variable names')
        state[stage] = result
    return final_graph(variant, state), state
