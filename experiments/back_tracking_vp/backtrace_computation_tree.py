#!/usr/bin/env python3
"""
Backtrace Computation Tree Visualization

Creates HTML visualizations of computation trees from vocab projection data,
showing how ground truth reasoning traces are (or are not) represented.

Reads results.json from analyze_gt_representation.py and creates interactive
HTML tables showing:
- Vocab projection grid (positions x ranks)
- Highlighted values: leaf operands (yellow), intermediate results (green), answer (blue)
- Lines connecting operands to their results
"""

import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from experiments.back_tracking_vp.solution_utils import to_number, extract_all_numbers


def get_position_tokens(num_latent: int, model_type: str, num_answer_tokens: int = 2) -> Tuple[List[str], List[str]]:
    """Generate input and output token names for each position.

    Args:
        num_latent: Number of latent tokens
        model_type: "coconut" or "codi"
        num_answer_tokens: Number of answer tokens (typically 2: answer + eos)

    Returns:
        Tuple of (input_tokens, output_tokens)
        - input_tokens: The token at each position (what goes IN to the model at that step)
        - output_tokens: The token predicted at each position (what comes OUT)
    """
    if model_type == "codi":
        # Input tokens (what's at each position)
        input_tokens = ["<|bot|>"]
        input_tokens += [f"<|latent|>_{i}" for i in range(num_latent)]
        input_tokens += ["<|eot|>", "The", "answer", "is", ":"]
        input_tokens += [f"ans_{i}" for i in range(num_answer_tokens)]

        # Output tokens (what's predicted at each position)
        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|eot|>", "The", "answer", "is", ":"]
        output_tokens += [f"ans_{i}" for i in range(num_answer_tokens)]
        output_tokens += ["<eos>"]
    else:  # coconut
        # Input tokens
        input_tokens = ["<|start|>"]
        input_tokens += [f"<|latent|>_{i}" for i in range(num_latent)]
        input_tokens += ["<|end|>", "###"]
        input_tokens += [f"ans_{i}" for i in range(num_answer_tokens)]

        # Output tokens
        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|end|>", "###"]
        output_tokens += [f"ans_{i}" for i in range(num_answer_tokens)]
        output_tokens += ["<eos>"]

    return input_tokens, output_tokens


# Colors for different computation steps
STEP_COLORS = [
    "#c62828",  # Red
    "#1565c0",  # Blue
    "#2e7d32",  # Green
    "#6a1b9a",  # Purple
    "#ef6c00",  # Orange
    "#00838f",  # Cyan
    "#ad1457",  # Pink
    "#4527a0",  # Deep purple
]


def parse_solution_values(solution_str: str) -> Tuple[set, set, Optional[Union[int, float]]]:
    """
    Extract leaf operands, intermediate results, and final answer from a solution string.

    Args:
        solution_str: Solution string with <<expr=result>> format

    Returns:
        Tuple of (leaf_operands, intermediate_results, final_answer)
        Values are int for whole numbers, float for non-trivial decimals.
    """
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)

    if not matches:
        return set(), set(), None

    all_operands = set()
    all_results = []
    identity_results = set()

    for expr, result_str in matches:
        result = to_number(result_str)
        operands = extract_all_numbers(expr)
        all_results.append(result)
        all_operands.update(operands)
        # Identity step (e.g., <<40=40>>): single operand equals result
        if len(operands) == 1 and operands[0] == result:
            identity_results.add(result)

    # Final answer is the last result
    final_answer = all_results[-1] if all_results else None

    # Intermediate results are all results except the final one,
    # excluding identity steps (no real computation produced them)
    intermediate_results = set(all_results[:-1]) - identity_results if len(all_results) > 1 else set()

    # Leaf operands are operands that are NOT intermediate results
    leaf_operands = all_operands - intermediate_results

    return leaf_operands, intermediate_results, final_answer


def extract_number_from_token(token: str) -> Optional[Union[int, float]]:
    """Extract numeric value from a token string, or None if not a number.

    Recognizes integers (16) and decimals (0.5, .5, 16.00).
    Returns int for whole numbers, float for non-trivial decimals.
    Only returns positive values.
    """
    cleaned = token.replace('\u0120', ' ').strip()
    if re.match(r'^-?(?:\d+(?:\.\d+)?|\.\d+)$', cleaned):
        val = to_number(cleaned)
        if val is not None:
            val = abs(val)
            if val > 0:
                return val
    return None


def _get_representative_int(value):
    """For multi-token numbers (like 0.5), get the first non-zero integer part.

    Mirrors _first_nonzero_int_token from analyze_gt_representation.py
    but works from the numeric value without needing a tokenizer.

    Examples: 0.5 → 5, 3.5 → 3, 0.25 → 25
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        s = f"{value:g}"
        parts = s.replace('-', '').split('.')
        for part in parts:
            if part:
                try:
                    v = int(part)
                    if v != 0:
                        return v
                except ValueError:
                    pass
    return None


def _build_display_entry(label, solution_str, best_tree, num_reasoning_positions):
    """Build a display entry dict for JavaScript solution switching."""
    leaf_operands, intermediate_results, final_answer = parse_solution_values(solution_str)

    # Build extended highlight sets
    leaf_highlight = set(leaf_operands)
    intermediate_highlight = set(intermediate_results)
    final_answer_highlight = set()
    if final_answer is not None:
        final_answer_highlight.add(final_answer)
    for val in leaf_operands:
        rep = _get_representative_int(val)
        if rep is not None and rep != val:
            leaf_highlight.add(rep)
    for val in intermediate_results:
        rep = _get_representative_int(val)
        if rep is not None and rep != val:
            intermediate_highlight.add(rep)
    if final_answer is not None:
        rep = _get_representative_int(final_answer)
        if rep is not None and rep != final_answer:
            final_answer_highlight.add(rep)

    # Parse step results for edge coloring
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)
    step_results = [to_number(m[1]) for m in matches]

    # Build tree nodes (use actual positions/ranks from best_tree,
    # since baseline answers may not be rank 0 at the answer position)
    tree_nodes = {}
    if best_tree and best_tree.get('nodes'):
        for node in best_tree['nodes']:
            if node.get('type') == 'question':
                key = f"cell_q_{node['value']}"
            else:
                key = f"cell_{node['position']}_{node['rank']}"
            tree_nodes[key] = node['type']

    # Build edges (use actual positions/ranks from best_tree)
    edges = []
    if best_tree and best_tree.get('edges'):
        for edge in best_tree['edges']:
            if edge['from_position'] == -1:
                from_id = f"cell_q_{edge['from_value']}"
            else:
                from_id = f"cell_{edge['from_position']}_{edge['from_rank']}"
            to_id = f"cell_{edge['to_position']}_{edge['to_rank']}"

            step_idx = 0
            for i, result in enumerate(step_results):
                if edge['to_value'] == result:
                    step_idx = i
                    break

            color = STEP_COLORS[step_idx % len(STEP_COLORS)]
            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'from_val': edge['from_value'],
                'to_val': edge['to_value'],
                'color': color
            })

    return {
        'label': label,
        'solution_str': solution_str,
        'edges': edges,
        'treeNodes': tree_nodes,
        'leafHighlight': sorted(leaf_highlight),
        'intermediateHighlight': sorted(intermediate_highlight),
        'finalAnswerHighlight': sorted(final_answer_highlight),
    }


def create_visualization_html(
    sample: Dict,
    output_path: Path,
    top_k: int = 10,
    model_type: str = "codi",
    num_latent: int = 6,
    force_answer_rank0: bool = True
) -> bool:
    """
    Create HTML visualization for a single sample with computation tree lines.

    Args:
        sample: Sample dict from results.json
        output_path: Path to save HTML file
        top_k: Number of top tokens shown
        model_type: "coconut" or "codi"
        num_latent: Number of latent tokens

    Returns:
        True if successful, False otherwise
    """
    vocab_projection = sample.get('vocab_projection_top_k', [])
    if not vocab_projection:
        logging.warning(f"Sample {sample['sample_idx']}: No vocab projection data")
        return False

    num_reasoning_positions = sample.get('num_reasoning_positions', len(vocab_projection))
    num_positions = len(vocab_projection)

    # Calculate number of answer tokens (positions after reasoning + delimiter)
    # For CODI: reasoning(8) + delimiter(4) + answer tokens
    # For coconut: reasoning(8) + ### + answer tokens
    if model_type == "codi":
        num_delimiter_tokens = 4  # The, answer, is, :
    else:
        num_delimiter_tokens = 1  # ###
    num_answer_tokens = num_positions - num_reasoning_positions - num_delimiter_tokens
    if num_answer_tokens < 1:
        num_answer_tokens = 2  # Default

    # Get input/output token names
    input_tokens, output_tokens = get_position_tokens(num_latent, model_type, num_answer_tokens)

    # Trim to actual number of positions
    input_tokens = input_tokens[:num_positions]
    output_tokens = output_tokens[:num_positions]

    # Get the best represented solution (or primary if none found)
    solutions = sample.get('solutions', [])
    if not solutions:
        logging.warning(f"Sample {sample['sample_idx']}: No solutions")
        return False

    # Find the best represented solution
    best_idx = sample.get('best_represented_idx')
    if best_idx is not None:
        solution = solutions[best_idx]
        solution_str = solution['solution']
        solution_found = True
        best_tree = solution.get('best_tree')
        is_primary = solution.get('is_primary', best_idx == 0)
    else:
        solution = solutions[0]  # Use primary
        solution_str = solution['solution']
        solution_found = False
        best_tree = None
        is_primary = True

    # Determine solution source label
    solution_source = "primary" if is_primary else "multichain"

    # Extract values from the solution
    leaf_operands, intermediate_results, final_answer = parse_solution_values(solution_str)

    # Build extended highlight sets for multi-token numbers.
    # E.g., 0.5 tokenizes as "0", ".", "5" → representative token is "5" → highlight 5 as leaf.
    leaf_highlight = set(leaf_operands)
    intermediate_highlight = set(intermediate_results)
    final_answer_highlight = set()
    if final_answer is not None:
        final_answer_highlight.add(final_answer)
    for val in leaf_operands:
        rep = _get_representative_int(val)
        if rep is not None and rep != val:
            leaf_highlight.add(rep)
    for val in intermediate_results:
        rep = _get_representative_int(val)
        if rep is not None and rep != val:
            intermediate_highlight.add(rep)
    if final_answer is not None:
        rep = _get_representative_int(final_answer)
        if rep is not None and rep != final_answer:
            final_answer_highlight.add(rep)

    # Find the first answer position (first position after delimiter where answer appears)
    # Calculate based on actual position structure:
    # - CODI: latents (num_latent) + eot (1) + "The answer is:" (4) = num_latent + 5
    # - Coconut: latents (num_latent) + end (1) + ### (1) = num_latent + 2
    if model_type == "codi":
        first_answer_position_idx = num_latent + 5
    else:
        first_answer_position_idx = num_latent + 2

    # Build cell data with highlighting
    cells_data = []

    for pos_idx, tokens in enumerate(vocab_projection):
        position_cells = []
        for rank, token in enumerate(tokens[:top_k]):
            int_val = extract_number_from_token(token)
            cell_id = f"cell_{pos_idx}_{rank}"

            # Determine cell type for highlighting
            cell_type = 'other'
            if int_val is not None:
                if int_val in final_answer_highlight and pos_idx >= num_reasoning_positions:
                    cell_type = 'answer'
                elif int_val in intermediate_highlight:
                    cell_type = 'intermediate'
                elif int_val in leaf_highlight:
                    cell_type = 'leaf'
                else:
                    cell_type = 'integer'

            position_cells.append({
                'token': token,
                'int_val': int_val,
                'cell_type': cell_type,
                'rank': rank,
                'cell_id': cell_id
            })
        cells_data.append(position_cells)

    # Parse solution to get step information for coloring edges
    pattern = r'<<(.+?)=(-?\d+(?:\.\d+)?)>>'
    matches = re.findall(pattern, solution_str)
    step_results = [to_number(m[1]) for m in matches]  # Results of each step

    # Build edges data for line drawing with step-based colors
    edges = []
    if best_tree and best_tree.get('edges'):
        for edge in best_tree['edges']:
            from_val = edge['from_value']
            from_pos = edge['from_position']
            from_rank = edge['from_rank']
            to_val = edge['to_value']
            to_pos = edge['to_position']
            to_rank = edge['to_rank']

            if from_pos == -1:
                from_id = f"cell_q_{from_val}"
            else:
                from_id = f"cell_{from_pos}_{from_rank}"
            to_id = f"cell_{to_pos}_{to_rank}"

            # Determine which step this edge belongs to based on to_val (the result)
            step_idx = 0
            for i, result in enumerate(step_results):
                if to_val == result:
                    step_idx = i
                    break

            color = STEP_COLORS[step_idx % len(STEP_COLORS)]

            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'from_val': from_val,
                'to_val': to_val,
                'color': color
            })

    # Build nodes data for highlighting the specific cells in the tree
    # Only include the first answer position node at rank 0 (top token), not all answer cells
    tree_nodes = {}
    if best_tree and best_tree.get('nodes'):
        for node in best_tree['nodes']:
            node_type = node['type']
            node_pos = node['position']

            # Question-sourced nodes: use value-based cell ID
            if node_type == 'question':
                key = f"cell_q_{node['value']}"
                tree_nodes[key] = node_type
                continue

            # For 'final' type nodes, use rank 0 by default (correct predictions)
            # or the actual rank from the tree (incorrect predictions)
            if node_type == 'final':
                if force_answer_rank0:
                    key = f"cell_{first_answer_position_idx}_0"
                else:
                    key = f"cell_{node_pos}_{node['rank']}"
                tree_nodes[key] = node_type
                continue

            key = f"cell_{node_pos}_{node['rank']}"
            tree_nodes[key] = node_type

    # For correct predictions, redirect final answer edges to rank 0
    if force_answer_rank0 and edges:
        for edge in edges:
            if edge['to_val'] == final_answer:
                edge['to_id'] = f"cell_{first_answer_position_idx}_0"

    # Build display entries for JS solution switching
    display_entries = []

    # GT entry (from already-computed data)
    display_entries.append({
        'label': f'GT ({solution_source})',
        'solution_str': solution_str,
        'edges': edges,
        'treeNodes': tree_nodes,
        'leafHighlight': sorted(leaf_highlight),
        'intermediateHighlight': sorted(intermediate_highlight),
        'finalAnswerHighlight': sorted(final_answer_highlight),
    })

    # Baseline entries (only found ones with trees get display entries)
    baselines = sample.get('baseline', {}).get('baselines', [])
    for i, bl in enumerate(baselines):
        if bl.get('times_found', 0) > 0 and bl.get('best_tree'):
            bl_entry = _build_display_entry(
                f'Baseline {i+1}',
                bl['solution'], bl['best_tree'],
                num_reasoning_positions
            )
            display_entries.append(bl_entry)

    # Build cell numeric values for JS
    cell_values = []
    for pos_cells in cells_data:
        cell_values.append([cell['int_val'] for cell in pos_cells])

    # Build baseline list HTML (always shown if baselines exist)
    baseline_list_html = ''
    if baselines:
        # Map found baseline solutions to their display_entries index
        found_baseline_indices = {}  # baseline list index -> display_entries index
        de_idx = 1  # display_entries[0] is GT
        for i, bl in enumerate(baselines):
            if bl.get('times_found', 0) > 0 and bl.get('best_tree'):
                found_baseline_indices[i] = de_idx
                de_idx += 1

        baseline_rows = ''
        for i, bl in enumerate(baselines):
            sol_escaped = bl['solution'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            times_found = bl.get('times_found', 0)
            found = times_found > 0 and bl.get('best_tree')
            status_cls = 'bl-found' if found else 'bl-not-found'
            status_text = f'Found ({times_found}x)' if found else 'Not found'

            if found:
                de_i = found_baseline_indices[i]
                action = f'<a href="#" class="bl-display-link" onclick="switchSolution({de_i}); return false;">Display</a>'
            else:
                action = ''

            baseline_rows += f"""                        <tr class="{status_cls}">
                            <td>{i+1}</td>
                            <td class="bl-solution"><code>{sol_escaped}</code></td>
                            <td class="bl-status">{status_text}</td>
                            <td>{action}</td>
                        </tr>
"""

        baseline_list_html = f"""
            <div class="baseline-section">
                <strong>Baselines:</strong>
                <a href="#" class="bl-display-link" onclick="switchSolution(0); return false;">[Show GT]</a>
                <table class="baseline-table">
                    <thead>
                        <tr><th>#</th><th>Solution</th><th>Status</th><th></th></tr>
                    </thead>
                    <tbody>
{baseline_rows}                    </tbody>
                </table>
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Vocab Projection - Sample {sample['sample_idx']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 20px;
        }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .status.found {{
            background-color: #4CAF50;
            color: white;
        }}
        .status.not-found {{
            background-color: #f44336;
            color: white;
        }}
        .question {{
            background-color: #e3f2fd;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
        }}
        .solution {{
            background-color: #fff3e0;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
        }}
        .info-row {{
            display: flex;
            gap: 15px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .info-item {{
            background-color: #f5f5f5;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
        }}
        .table-container {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
            position: relative;
        }}
        .table-wrapper {{
            position: relative;
        }}
        table {{
            border-collapse: separate;
            border-spacing: 2px;
            position: relative;
            z-index: 1;
        }}
        th {{
            background-color: #555;
            color: white;
            padding: 6px 10px;
            text-align: center;
            font-size: 10px;
            white-space: nowrap;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 4px 8px;
            text-align: center;
            font-family: monospace;
            font-size: 11px;
            position: relative;
        }}
        .rank-col {{
            background-color: #f5f5f5;
            font-weight: bold;
            width: 40px;
        }}
        .input-row td {{
            background-color: #555;
            color: white;
            font-size: 10px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        .input-row .rank-col {{
            background-color: #555;
            color: white;
        }}
        /* Cell type highlighting */
        .cell-leaf {{
            background-color: #FFF59D;
            font-weight: bold;
        }}
        .cell-intermediate {{
            background-color: #A5D6A7;
            font-weight: bold;
        }}
        .cell-answer {{
            background-color: #90CAF9;
            font-weight: bold;
        }}
        .cell-question {{
            background-color: #FFE0B2;
            font-weight: bold;
        }}
        .cell-integer {{
            background-color: #fff;
        }}
        .cell-other {{
            background-color: #f9f9f9;
            color: #999;
        }}
        /* Tree node highlighting (boxed) */
        .tree-node {{
            outline: 3px solid #333;
            outline-offset: -1px;
        }}
        .question-numbers-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .question-numbers-row .qn-label {{
            font-weight: bold;
            color: #555;
            white-space: nowrap;
        }}
        .question-numbers-row .qn-cell {{
            background-color: #FFE0B2;
            border: 1px solid #ddd;
            padding: 4px 8px;
            font-family: monospace;
            font-size: 11px;
            text-align: center;
            border-radius: 3px;
        }}
        .question-numbers-row .qn-cell.tree-node {{
            outline: 3px solid #333;
            outline-offset: -1px;
        }}
        /* SVG overlay for lines */
        .svg-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 2;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        .legend-color.boxed {{
            outline: 2px solid #333;
            outline-offset: -1px;
        }}
        .baseline-section {{
            margin: 12px 0 0 0;
        }}
        .baseline-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 6px;
            font-size: 13px;
        }}
        .baseline-table th {{
            text-align: left;
            padding: 4px 8px;
            border-bottom: 2px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
        .baseline-table td {{
            padding: 4px 8px;
            border-bottom: 1px solid #eee;
        }}
        .baseline-table .bl-solution {{
            font-family: monospace;
            font-size: 12px;
        }}
        .baseline-table .bl-status {{
            font-weight: bold;
            font-size: 12px;
        }}
        .bl-found .bl-status {{
            color: #4CAF50;
        }}
        .bl-not-found .bl-status {{
            color: #999;
        }}
        .bl-display-link {{
            color: #1565c0;
            text-decoration: none;
            font-size: 12px;
            font-weight: bold;
        }}
        .bl-display-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Vocabulary Projections (Top {top_k} tokens per position)</h1>
            <span class="status {'found' if solution_found else 'not-found'}">
                {'GT Solution FOUND' if solution_found else 'GT Solution NOT FOUND'}
            </span>

            <div class="question">
                <strong>Question:</strong> {sample.get('question', 'N/A')}
            </div>

            <div class="solution" id="solutionText">
                <strong>Solution ({solution_source}):</strong> {solution_str}
            </div>
            {baseline_list_html}

            <div class="info-row">
                <div class="info-item"><strong>GT Answer:</strong> {sample.get('gt_answer', 'N/A')}</div>
                <div class="info-item"><strong>Model Answer:</strong> {sample.get('model_answer', 'N/A')}</div>
                <div class="info-item"><strong>Correct:</strong> {'Yes' if sample.get('answer_correct') else 'No'}</div>
                <div class="info-item"><strong>Steps:</strong> {sample.get('step_count', 'N/A')}</div>
            </div>
        </div>

        <div class="table-container">
            <div class="table-wrapper" id="tableWrapper">
"""

    # Add question numbers row if any display entry uses question-sourced nodes
    question_values_in_trees = set()
    for entry in display_entries:
        for key in entry['treeNodes']:
            if key.startswith('cell_q_'):
                question_values_in_trees.add(key[len('cell_q_'):])

    if question_values_in_trees:
        html += '                <div class="question-numbers-row" id="questionNumbersRow">\n'
        html += '                    <span class="qn-label">Question Numbers:</span>\n'
        for qval in sorted(question_values_in_trees, key=lambda x: float(x)):
            cell_id = f"cell_q_{qval}"
            # Check if this value is in the current (initial) tree
            is_tree_node = cell_id in tree_nodes
            cls = "qn-cell tree-node" if is_tree_node else "qn-cell"
            html += f'                    <span id="{cell_id}" class="{cls}">{qval}</span>\n'
        html += '                </div>\n'

    html += """                <table id="vocabTable">
                    <thead>
                        <tr>
                            <th>Output</th>
"""

    # Add output token headers
    for pos_idx, out_token in enumerate(output_tokens):
        out_token_escaped = out_token.replace('<', '&lt;').replace('>', '&gt;')
        html += f'                            <th>{out_token_escaped}</th>\n'

    html += """                        </tr>
                    </thead>
                    <tbody>
"""

    # Add rows for each rank
    for rank in range(top_k):
        html += f"                        <tr>\n"
        html += f"                            <td class=\"rank-col\">{rank + 1}</td>\n"

        for pos_idx, position_cells in enumerate(cells_data):
            if rank < len(position_cells):
                cell = position_cells[rank]
                token_display = cell['token'].replace('<', '&lt;').replace('>', '&gt;')
                cell_id = cell['cell_id']
                cell_type = cell['cell_type']

                # Override cell_type for cells in the best tree
                if cell_id in tree_nodes:
                    node_type = tree_nodes[cell_id]
                    if node_type == 'leaf':
                        cell_type = 'leaf'
                    elif node_type == 'intermediate':
                        cell_type = 'intermediate'
                    elif node_type == 'final':
                        cell_type = 'answer'

                css_class = f"cell-{cell_type}"

                # Add tree-node class if this cell is in the best tree
                if cell_id in tree_nodes:
                    css_class += " tree-node"

                html += f'                            <td id="{cell_id}" class="{css_class}">{token_display}</td>\n'
            else:
                html += f'                            <td class="cell-other">-</td>\n'

        html += f"                        </tr>\n"

    # Add bottom row with input tokens (same styling as header)
    html += f"                        <tr class=\"input-row\">\n"
    html += f"                            <td class=\"rank-col\">Input</td>\n"
    for pos_idx, in_token in enumerate(input_tokens):
        in_token_escaped = in_token.replace('<', '&lt;').replace('>', '&gt;')
        html += f'                            <td>{in_token_escaped}</td>\n'
    html += f"                        </tr>\n"

    html += """                    </tbody>
                </table>
                <svg class="svg-overlay" id="svgOverlay">
                </svg>
            </div>

            <div class="legend">
                <strong>Legend:</strong>
                <div class="legend-item">
                    <div class="legend-color boxed" style="background-color: #FFF59D;"></div>
                    <span>Leaf Operand</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color boxed" style="background-color: #FFE0B2;"></div>
                    <span>Question Operand</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color boxed" style="background-color: #A5D6A7;"></div>
                    <span>Intermediate Result</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color boxed" style="background-color: #90CAF9;"></div>
                    <span>Final Answer</span>
                </div>
"""

    # Add step color legend with arrows
    num_steps = len(step_results)
    for i in range(num_steps):
        color = STEP_COLORS[i % len(STEP_COLORS)]
        html += f"""                <div class="legend-item">
                    <svg width="30" height="20" style="vertical-align: middle;">
                        <defs>
                            <marker id="legend-arrow-{i}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="{color}" />
                            </marker>
                        </defs>
                        <line x1="2" y1="10" x2="22" y2="10" stroke="{color}" stroke-width="2" marker-end="url(#legend-arrow-{i})" />
                    </svg>
                    <span>Step {i + 1}</span>
                </div>
"""

    html += """            </div>
        </div>
    </div>

    <script>
"""

    # Add display entries and cell values for JavaScript
    html += f"        const displayEntries = {json.dumps(display_entries)};\n"
    html += f"        const cellValues = {json.dumps(cell_values)};\n"
    html += f"        const numPositions = {num_positions};\n"
    html += f"        const topK = {top_k};\n"
    html += f"        const numReasoningPositions = {num_reasoning_positions};\n"

    html += """
        let currentDisplayIdx = 0;

        function switchSolution(idx) {
            if (idx === undefined) idx = 0;
            currentDisplayIdx = idx;
            const entry = displayEntries[idx];

            // Update solution text
            document.getElementById('solutionText').innerHTML =
                '<strong>Solution (' + entry.label + '):</strong> ' + entry.solution_str;

            // Re-highlight all cells
            for (let pos = 0; pos < numPositions; pos++) {
                for (let rank = 0; rank < topK; rank++) {
                    const cellId = 'cell_' + pos + '_' + rank;
                    const cell = document.getElementById(cellId);
                    if (!cell) continue;

                    const val = cellValues[pos][rank];
                    let cellType = 'other';

                    if (val !== null) {
                        if (entry.finalAnswerHighlight.includes(val) && pos >= numReasoningPositions) {
                            cellType = 'answer';
                        } else if (entry.intermediateHighlight.includes(val)) {
                            cellType = 'intermediate';
                        } else if (entry.leafHighlight.includes(val)) {
                            cellType = 'leaf';
                        } else {
                            cellType = 'integer';
                        }
                    }

                    // Override for tree nodes
                    const nodeType = entry.treeNodes[cellId];
                    if (nodeType) {
                        if (nodeType === 'leaf') cellType = 'leaf';
                        else if (nodeType === 'intermediate') cellType = 'intermediate';
                        else if (nodeType === 'final') cellType = 'answer';
                        else if (nodeType === 'question') cellType = 'question';
                    }

                    let cls = 'cell-' + cellType;
                    if (nodeType) cls += ' tree-node';
                    cell.className = cls;
                }
            }

            // Update question number cells (highlight/unhighlight based on current entry)
            const qnRow = document.getElementById('questionNumbersRow');
            if (qnRow) {
                const qnCells = qnRow.querySelectorAll('.qn-cell');
                qnCells.forEach(function(cell) {
                    const nodeType = entry.treeNodes[cell.id];
                    if (nodeType === 'question') {
                        cell.className = 'qn-cell tree-node';
                    } else {
                        cell.className = 'qn-cell';
                    }
                });
            }

            // Redraw edges
            drawLines(entry.edges);
        }

        function drawLines(edgesData) {
            const svg = document.getElementById('svgOverlay');
            const wrapper = document.getElementById('tableWrapper');
            const table = document.getElementById('vocabTable');

            svg.style.width = table.offsetWidth + 'px';
            svg.style.height = table.offsetHeight + 'px';
            svg.setAttribute('width', table.offsetWidth);
            svg.setAttribute('height', table.offsetHeight);

            while (svg.childNodes.length > 0) {
                svg.removeChild(svg.lastChild);
            }

            if (!edgesData || edgesData.length === 0) return;

            const colors = [...new Set(edgesData.map(e => e.color))];
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            colors.forEach(color => {
                const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                const colorId = color.replace('#', '');
                marker.setAttribute('id', 'arrowhead-' + colorId);
                marker.setAttribute('markerWidth', '10');
                marker.setAttribute('markerHeight', '7');
                marker.setAttribute('refX', '9');
                marker.setAttribute('refY', '3.5');
                marker.setAttribute('orient', 'auto');
                const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
                polygon.setAttribute('fill', color);
                marker.appendChild(polygon);
                defs.appendChild(marker);
            });
            svg.appendChild(defs);

            edgesData.forEach(edge => {
                const fromCell = document.getElementById(edge.from_id);
                const toCell = document.getElementById(edge.to_id);

                if (fromCell && toCell) {
                    const fromRect = fromCell.getBoundingClientRect();
                    const toRect = toCell.getBoundingClientRect();
                    const wrapperRect = wrapper.getBoundingClientRect();

                    const fromX = fromRect.left + fromRect.width / 2 - wrapperRect.left;
                    const fromY = fromRect.top + fromRect.height / 2 - wrapperRect.top;
                    const toX = toRect.left - wrapperRect.left + 5;
                    const toY = toRect.top + toRect.height / 2 - wrapperRect.top;

                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', fromX);
                    line.setAttribute('y1', fromY);
                    line.setAttribute('x2', toX);
                    line.setAttribute('y2', toY);
                    line.setAttribute('stroke', edge.color);
                    line.setAttribute('stroke-width', '2');
                    const colorId = edge.color.replace('#', '');
                    line.setAttribute('marker-end', 'url(#arrowhead-' + colorId + ')');

                    svg.appendChild(line);
                }
            });
        }

        // Draw initial edges on page load
        window.addEventListener('load', function() {
            drawLines(displayEntries[0].edges);
        });
        window.addEventListener('resize', function() {
            drawLines(displayEntries[currentDisplayIdx].edges);
        });
    </script>
</body>
</html>
"""

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return True


def create_index_html(
    results: List[Dict],
    output_dir: Path,
    found_samples: List[int],
    not_found_samples: List[int]
):
    """Create an index.html linking to all visualizations."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Vocab Projection Visualizations - Index</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
        }
        .section {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .found h2 {
            color: #4CAF50;
        }
        .not-found h2 {
            color: #f44336;
        }
        .sample-link {
            display: block;
            padding: 12px 16px;
            margin: 8px 0;
            background-color: #f9f9f9;
            border-radius: 4px;
            text-decoration: none;
            color: #333;
            transition: background-color 0.2s;
        }
        .sample-link:hover {
            background-color: #e8e8e8;
        }
        .sample-link .idx {
            font-weight: bold;
            margin-right: 10px;
        }
        .sample-link .answer {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vocab Projection Visualizations</h1>

        <div class="section found">
            <h2>Solution Found (first 5)</h2>
"""

    for sample_idx in found_samples:
        sample = next((s for s in results if s['sample_idx'] == sample_idx), None)
        if sample:
            html += f"""            <a href="sample_{sample_idx:03d}_GT_found.html" class="sample-link">
                <span class="idx">Sample {sample_idx}</span>
                <span class="answer">GT: {sample.get('gt_answer', 'N/A')} | Model: {sample.get('model_answer', 'N/A')}</span>
            </a>
"""

    html += """        </div>

        <div class="section not-found">
            <h2>Solution NOT Found (first 5)</h2>
"""

    for sample_idx in not_found_samples:
        sample = next((s for s in results if s['sample_idx'] == sample_idx), None)
        if sample:
            html += f"""            <a href="sample_{sample_idx:03d}_GT_not_found.html" class="sample-link">
                <span class="idx">Sample {sample_idx}</span>
                <span class="answer">GT: {sample.get('gt_answer', 'N/A')} | Model: {sample.get('model_answer', 'N/A')}</span>
            </a>
"""

    html += """        </div>
    </div>
</body>
</html>
"""

    index_path = output_dir / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logging.info(f"Created index at {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create HTML visualizations of vocab projection computation trees'
    )

    parser.add_argument(
        '--results_json',
        type=str,
        required=True,
        help='Path to results.json from analyze_gt_representation.py'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for HTML files (default: same as results.json)'
    )

    parser.add_argument(
        '--num_found',
        type=int,
        default=5,
        help='Number of "found" examples to visualize'
    )

    parser.add_argument(
        '--num_not_found',
        type=int,
        default=5,
        help='Number of "not found" examples to visualize'
    )

    parser.add_argument(
        '--sample_indices',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample indices to visualize (overrides num_found/num_not_found)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load results
    results_path = Path(args.results_json)
    logging.info(f"Loading results from {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    per_sample = data.get('per_sample', [])
    metadata = data.get('metadata', {})
    logging.info(f"Loaded {len(per_sample)} samples")

    # Extract model_type and num_latent from metadata
    model_type = metadata.get('model_type', 'codi')
    num_latent = metadata.get('num_latent', 6)
    top_k = metadata.get('top_k', 10)
    logging.info(f"Model type: {model_type}, num_latent: {num_latent}, top_k: {top_k}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Determine which samples to visualize
    if args.sample_indices:
        # Explicit sample indices provided — use them directly
        all_samples_to_viz = args.sample_indices
        found_samples = [idx for idx in all_samples_to_viz
                         if any(s['sample_idx'] == idx and s.get('any_gt_found') for s in per_sample)]
        not_found_samples = [idx for idx in all_samples_to_viz if idx not in found_samples]
    else:
        # Default: pick first N found/not-found among correct answers
        found_samples = []
        not_found_samples = []

        for sample in per_sample:
            if not sample.get('answer_correct'):
                continue

            if sample.get('any_gt_found'):
                found_samples.append(sample['sample_idx'])
            else:
                not_found_samples.append(sample['sample_idx'])

        found_samples = found_samples[:args.num_found]
        not_found_samples = not_found_samples[:args.num_not_found]
        all_samples_to_viz = found_samples + not_found_samples

    logging.info(f"Found samples to visualize: {found_samples}")
    logging.info(f"Not found samples to visualize: {not_found_samples}")
    success_count = 0

    for sample_idx in all_samples_to_viz:
        sample = next((s for s in per_sample if s['sample_idx'] == sample_idx), None)
        if sample is None:
            logging.warning(f"Sample {sample_idx} not found in results")
            continue

        # Determine if GT was found for this sample
        gt_status = "GT_found" if sample.get('any_gt_found') else "GT_not_found"
        output_path = output_dir / f"sample_{sample_idx:03d}_{gt_status}.html"

        if create_visualization_html(sample, output_path, top_k=top_k, model_type=model_type, num_latent=num_latent):
            logging.info(f"Created visualization for sample {sample_idx}")
            success_count += 1
        else:
            logging.warning(f"Failed to create visualization for sample {sample_idx}")

    # Create index
    create_index_html(per_sample, output_dir, found_samples, not_found_samples)

    logging.info(f"\nCreated {success_count} visualizations in {output_dir}")
    logging.info(f"Open {output_dir / 'index.html'} to browse")


if __name__ == "__main__":
    main()
