#!/usr/bin/env python3
"""
Forward Chaining Visualization

Creates HTML visualizations of computation trees discovered through forward chaining.
Shows vocab projection table with arrows connecting operands to results.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def extract_number_from_token(token: str) -> Optional[int]:
    """Extract integer value from a token string, or None if not a number."""
    cleaned = token.replace('\u0120', ' ').strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def get_tree_status_badge(sample: Dict, metadata: Dict = None) -> str:
    """
    Compute tree verification status and return HTML badge.

    Args:
        sample: Sample dict with validation info
        metadata: Optional metadata dict with validation settings

    Returns:
        HTML string for the tree status badge (Verified, Partially Verified, or Unverified)
    """
    validation_info = sample.get('validation')
    best_tree = sample.get('best_tree')

    # If no tree found, no status badge
    if not best_tree or not best_tree.get('steps'):
        return ''

    # If no validation info, no status badge
    if not validation_info or 'steps' not in validation_info:
        return ''

    # Count verified and total steps in the tree
    tree_step_positions = {s['position_result'] for s in best_tree.get('steps', [])}
    verified_count = 0
    total_count = len(tree_step_positions)

    for v_step in validation_info.get('steps', []):
        if v_step.get('result_position') in tree_step_positions:
            if v_step.get('verified', False):
                verified_count += 1

    # Get validation settings from metadata
    validation_n = metadata.get('validation_n', 3) if metadata else 3
    required_passes = metadata.get('validation_required_passes', 2) if metadata else 2
    max_rank = metadata.get('validation_max_rank', 1) if metadata else 1
    settings_str = f" ({required_passes}/{validation_n} passes, top-{max_rank})"

    # Determine status
    if verified_count == total_count and total_count > 0:
        status_class = 'verified'
        status_text = f'Tree Status: VERIFIED{settings_str}'
    elif verified_count > 0:
        status_class = 'partially-verified'
        status_text = f'Tree Status: PARTIALLY VERIFIED ({verified_count}/{total_count}){settings_str}'
    else:
        status_class = 'unverified'
        status_text = f'Tree Status: UNVERIFIED{settings_str}'

    return f'<span class="status {status_class}">{status_text}</span>'


def get_position_tokens(num_latent: int, model_type: str, num_extra: int = 4) -> Tuple[List[str], List[str]]:
    """Generate input and output token names for each position.

    num_extra: Number of extra positions after the delimiter for multi-token answers.
               Default is 4 (ans_0, ans_1, ans_2, ans_3) to handle answers up to 3 tokens
               plus the <|endoftext|> position.
    """
    if model_type == "codi":
        input_tokens = ["<|bot|>"]
        input_tokens += [f"<|latent|>_{i}" for i in range(num_latent)]
        input_tokens += ["<|eot|>", "The", "answer", "is", ":"]
        input_tokens += [f"ans_{i}" for i in range(num_extra)]

        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|eot|>", "The", "answer", "is", ":"]
        output_tokens += [f"ans_{i}" for i in range(num_extra)]
        output_tokens += ["<eos>"]
    else:  # coconut
        input_tokens = ["<|start|>"]
        input_tokens += [f"<|latent|>_{i}" for i in range(num_latent)]
        input_tokens += ["<|end|>", "###"]
        input_tokens += [f"ans_{i}" for i in range(num_extra)]

        output_tokens = [f"<|latent|>_{i}" for i in range(num_latent)]
        output_tokens += ["<|end|>", "###"]
        output_tokens += [f"ans_{i}" for i in range(num_extra)]
        output_tokens += ["<eos>"]

    return input_tokens, output_tokens


def create_visualization_html(
    sample: Dict,
    output_path: Path,
    top_k: int = 10,
    model_type: str = "coconut",
    num_latent: int = 6,
    metadata: Dict = None
) -> bool:
    """Create HTML visualization for a single sample."""
    vocab_projection = sample.get('vocab_projection_top_k', [])
    if not vocab_projection:
        logging.warning(f"Sample {sample['sample_idx']}: No vocab projection data")
        return False

    num_positions = len(vocab_projection)
    best_tree = sample.get('best_tree')
    all_steps = sample.get('all_best_steps', [])
    tree_found = sample.get('tree_found', False)

    # Get input/output token names
    input_tokens, output_tokens = get_position_tokens(num_latent, model_type, num_positions - num_latent - 2)
    input_tokens = input_tokens[:num_positions]
    output_tokens = output_tokens[:num_positions]

    # Collect values used in tree for highlighting
    tree_values = set()
    leaf_values = set()
    intermediate_values = set()
    final_value = None
    uses_direct_final = False
    answer_pos_idx = sample.get('answer_position_idx', num_latent + 2)
    # Offset from answer_pos_idx to the actual integer token (handles Llama's leading space)
    answer_token_offset = sample.get('answer_token_offset', 0)

    if best_tree:
        uses_direct_final = best_tree.get('uses_direct_final_step', False)
        for node in best_tree.get('nodes', []):
            # Skip "combined" position nodes (synthetic final answer)
            if node['position'] != "combined":
                tree_values.add((node['value'], node['position']))
            if node['type'] == 'leaf':
                leaf_values.add(node['value'])
            elif node['type'] == 'intermediate':
                intermediate_values.add(node['value'])
            elif node['type'] == 'final':
                final_value = node['value']
                # For direct final steps, also add the final answer at the answer position
                if uses_direct_final:
                    tree_values.add((node['value'], answer_pos_idx))

    # Also highlight final answer at answer position
    model_answer = sample.get('model_answer')

    # Check if include_question_tokens was used
    include_question_tokens = sample.get('include_question_tokens', False)
    question_numbers = sample.get('question_numbers', [])
    copy_from_question_steps = sample.get('copy_from_question_steps', [])

    # Actual position of the integer token for the answer (after skipping any leading spaces)
    actual_answer_pos = answer_pos_idx + answer_token_offset

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
                # Special case: final answer at actual answer token position (accounts for leading space)
                if int_val == final_value and pos_idx == actual_answer_pos and rank == 0:
                    cell_type = 'answer'
                # Check if this cell is part of the tree (must match both value AND position)
                elif (int_val, pos_idx) in tree_values:
                    # Find the node type
                    for node in best_tree.get('nodes', []) if best_tree else []:
                        if node['value'] == int_val and node['position'] == pos_idx:
                            if node['type'] == 'leaf':
                                cell_type = 'leaf'
                            elif node['type'] == 'intermediate':
                                cell_type = 'intermediate'
                            elif node['type'] == 'final':
                                # For direct final steps, the final node at computation position
                                # is actually intermediate (the 'answer' is at actual_answer_pos)
                                if uses_direct_final and pos_idx != actual_answer_pos:
                                    cell_type = 'intermediate'
                                else:
                                    cell_type = 'answer'
                            break
                elif int_val == model_answer and pos_idx == actual_answer_pos and rank == 0:
                    # Only mark as answer at the actual answer token position
                    # and only for top-1 token
                    cell_type = 'answer'
                else:
                    # It's an integer but not part of the tree
                    cell_type = 'integer'

            position_cells.append({
                'token': token,
                'int_val': int_val,
                'cell_type': cell_type,
                'rank': rank,
                'cell_id': cell_id
            })
        cells_data.append(position_cells)

    # Build edges data for line drawing
    # Include edges for: used steps in tree, unused steps (grey), and final combination
    edges = []
    used_step_positions = set()

    if best_tree:
        tree_steps = best_tree.get('steps', [])
        used_step_positions = {s['position_operands'] for s in tree_steps}
        final_combo = best_tree.get('final_combination')
        uses_direct_final = best_tree.get('uses_direct_final_step', False)
        final_answer = best_tree.get('final_answer')
        answer_pos_idx = sample.get('answer_position_idx', num_latent + 2)
        answer_token_offset = sample.get('answer_token_offset', 0)

        # Add edges from tree (used steps)
        for edge in best_tree.get('edges', []):
            from_pos = edge['from_position']
            from_rank = edge['from_rank']
            to_pos = edge['to_position']
            to_rank = edge['to_rank']

            # Handle "combined" position - draw arrows to final answer position
            if to_pos == "combined":
                # Draw to the actual integer token (accounting for leading space offset)
                actual_ans_pos = answer_pos_idx + answer_token_offset
                to_id = f"cell_{actual_ans_pos}_0"  # Top-1 at actual answer token
                to_pos = actual_ans_pos
                to_rank = 0
            else:
                # Draw to the actual result position (no redirection)
                to_id = f"cell_{to_pos}_{to_rank}"

            # Handle "question" source position - draw from question number cell
            if from_pos == "question":
                from_id = f"qnum_{edge['from_value']}"
            else:
                from_id = f"cell_{from_pos}_{from_rank}"

            # Determine step color based on which step this edge belongs to
            # Match by position, not just result value (avoids color collision when steps produce same result)
            step_idx = 0
            original_to_pos = edge.get('to_position')
            is_copy_edge = edge.get('is_copy', False)

            if is_copy_edge:
                # Copy edges get their own color (next after tree steps)
                step_idx = len(tree_steps)
            else:
                for i, step in enumerate(tree_steps):
                    # Match by the step's result position
                    if original_to_pos == step['position_result'] or \
                       (original_to_pos == "combined" and edge['from_position'] == step['position_result']):
                        step_idx = i
                        break

                # Final combination gets its own color (next color after tree steps)
                if original_to_pos == "combined":
                    step_idx = len(tree_steps)

            color = STEP_COLORS[step_idx % len(STEP_COLORS)]

            edges.append({
                'from_id': from_id,
                'to_id': to_id,
                'from_val': edge['from_value'],
                'to_val': edge['to_value'],
                'color': color,
                'used': True
            })

    # Add edges for unused steps (grey)
    UNUSED_COLOR = "#999999"
    copy_step = best_tree.get('copy_step') if best_tree else None
    for step in all_steps:
        if step['position_operands'] not in used_step_positions:
            pos_res = step['position_result']

            # Skip steps that produce the final answer at the answer position
            # when we have a copy step (since the copy handles that)
            if copy_step and step['result'] == copy_step['result'] and pos_res == copy_step['position_result']:
                continue

            # Edge from operand1 to result
            # If operand1 is from intermediate, use its source position
            if step.get('operand1_is_intermediate'):
                from_pos1 = step['operand1_source_pos']
                from_rank1 = step.get('operand1_ir_rank', 0)  # rank at source position
            else:
                from_pos1 = step['position_operands']
                from_rank1 = step['operand1_rank']

            # Handle question number source
            if from_pos1 == "question":
                from_id1 = f"qnum_{step['operand1']}"
            else:
                from_id1 = f"cell_{from_pos1}_{from_rank1}"

            edges.append({
                'from_id': from_id1,
                'to_id': f"cell_{pos_res}_{step['result_rank']}",
                'from_val': step['operand1'],
                'to_val': step['result'],
                'color': UNUSED_COLOR,
                'used': False
            })

            # Edge from operand2 to result
            # If operand2 is from intermediate, use its source position
            if step.get('operand2_is_intermediate'):
                from_pos2 = step['operand2_source_pos']
                from_rank2 = step.get('operand2_ir_rank', 0)  # rank at source position
            else:
                from_pos2 = step['position_operands']
                from_rank2 = step['operand2_rank']

            # Handle question number source
            if from_pos2 == "question":
                from_id2 = f"qnum_{step['operand2']}"
            else:
                from_id2 = f"cell_{from_pos2}_{from_rank2}"

            edges.append({
                'from_id': from_id2,
                'to_id': f"cell_{pos_res}_{step['result_rank']}",
                'from_val': step['operand2'],
                'to_val': step['result'],
                'color': UNUSED_COLOR,
                'used': False
            })

            # Edge from operand3 to result (for 3-operand steps)
            if step.get('operand3') is not None:
                if step.get('operand3_is_intermediate'):
                    from_pos3 = step['operand3_source_pos']
                    from_rank3 = step.get('operand3_ir_rank', 0)
                else:
                    from_pos3 = step['position_operands']
                    from_rank3 = step['operand3_rank']

                # Handle question number source
                if from_pos3 == "question":
                    from_id3 = f"qnum_{step['operand3']}"
                else:
                    from_id3 = f"cell_{from_pos3}_{from_rank3}"

                edges.append({
                    'from_id': from_id3,
                    'to_id': f"cell_{pos_res}_{step['result_rank']}",
                    'from_val': step['operand3'],
                    'to_val': step['result'],
                    'color': UNUSED_COLOR,
                    'used': False
                })

    # Add edges for copy_from_question steps (when include_question_tokens is True)
    COPY_FROM_QUESTION_COLOR = "#FF9800"  # Orange for copy from question
    for copy_step in copy_from_question_steps:
        from_id = f"qnum_{copy_step['operand1']}"
        to_id = f"cell_{copy_step['position_result']}_{copy_step['result_rank']}"
        edges.append({
            'from_id': from_id,
            'to_id': to_id,
            'from_val': copy_step['operand1'],
            'to_val': copy_step['result'],
            'color': COPY_FROM_QUESTION_COLOR,
            'used': True,
            'is_copy_from_question': True
        })

    # Build tree nodes for highlighting
    tree_nodes = {}
    if best_tree:
        uses_direct_final = best_tree.get('uses_direct_final_step', False)
        answer_pos_idx = sample.get('answer_position_idx', num_latent + 2)
        answer_token_offset = sample.get('answer_token_offset', 0)
        # Actual position of the integer token (after skipping any leading spaces)
        actual_ans_pos = answer_pos_idx + answer_token_offset

        for node in best_tree.get('nodes', []):
            # Skip "combined" position nodes (synthetic final answer)
            if node['position'] == "combined":
                # Add the final answer at the actual integer token position
                key = f"cell_{actual_ans_pos}_0"
                tree_nodes[key] = 'final'
                continue

            # For direct final step, redirect the final node to answer position
            if uses_direct_final and node['type'] == 'final':
                key = f"cell_{actual_ans_pos}_0"
                tree_nodes[key] = 'final'
            else:
                key = f"cell_{node['position']}_{node['rank']}"
                tree_nodes[key] = node['type']

    def format_3op_expression(o1, o2, o3, op, sep=""):
        """Format a 3-operand expression with the correct operators.

        Args:
            o1, o2, o3: The three operands
            op: Operation code (e.g., "+", "+-", "*/", "+*", etc.)
            sep: Separator between operators and operands (e.g., " " for legend, "" for compact)

        Returns:
            Formatted expression string (without result)
        """
        # Map operation codes to operator pairs
        op_map = {
            "+": ("+", "+"),      # a + b + c
            "-": ("-", "-"),      # not used but for completeness
            "*": ("*", "*"),      # a * b * c
            "+-": ("+", "-"),     # a + b - c
            "--": ("-", "-"),     # a - b - c
            "*/": ("*", "/"),     # a * b / c
            "//": ("/", "/"),     # a / b / c
            "+*": ("+", "*"),     # a + b * c (standard precedence: a + (b*c))
            "-*": ("-", "*"),     # a - b * c (standard precedence: a - (b*c))
            "*+": ("*", "+"),     # a * b + c (standard precedence: (a*b) + c)
            "*-": ("*", "-"),     # a * b - c (standard precedence: (a*b) - c)
            "+/": ("+", "/"),     # a + b / c (standard precedence: a + (b/c))
            "-/": ("-", "/"),     # a - b / c (standard precedence: a - (b/c))
            "/+": ("/", "+"),     # a / b + c (standard precedence: (a/b) + c)
            "/-": ("/", "-"),     # a / b - c (standard precedence: (a/b) - c)
        }
        if op in op_map:
            op1_str, op2_str = op_map[op]
            return f"{o1}{sep}{op1_str}{sep}{o2}{sep}{op2_str}{sep}{o3}"
        else:
            # Fallback: use the op string as-is between operands
            return f"{o1}{sep}{op}{sep}{o2}{sep}{op}{sep}{o3}"

    def format_step_expr(step):
        """Format a step as <<expr=result>> string, handling 2 and 3 operand steps."""
        op = step['operation']
        if step.get('operand3') is not None:
            expr = format_3op_expression(step['operand1'], step['operand2'], step['operand3'], op)
            return f"<<{expr}={step['result']}>>"
        else:
            return f"<<{step['operand1']}{op}{step['operand2']}={step['result']}>>"

    def format_combination_expr(values, op, result):
        """Format a combination expression with 2 or 3 operands."""
        if len(values) == 2:
            return f"<<{values[0]}{op}{values[1]}={result}>>"
        elif len(values) == 3:
            expr = format_3op_expression(values[0], values[1], values[2], op)
            return f"<<{expr}={result}>>"
        return ""

    def format_legend_step(step):
        """Format a step for legend display with proper operators for 3-operand steps."""
        op = step['operation']
        o1, o2, result = step['operand1'], step['operand2'], step['result']
        o3 = step.get('operand3')
        if o3 is not None:
            expr = format_3op_expression(o1, o2, o3, op, sep=" ")
            return f"{expr} = {result}"
        else:
            return f"{o1} {op} {o2} = {result}"

    # Build steps expression for display (using <<expr=result>> format like GT)
    steps_expr = []
    if best_tree:
        for step in best_tree.get('steps', []):
            steps_expr.append(format_step_expr(step))
        # Add final combination if present
        final_combo = best_tree.get('final_combination')
        if final_combo:
            values = final_combo['values']
            op = final_combo['operation']
            final_answer = best_tree.get('final_answer')
            steps_expr.append(format_combination_expr(values, op, final_answer))

    # Build unused steps expression for display
    unused_steps_expr = []
    copy_step_info = best_tree.get('copy_step') if best_tree else None
    for step in all_steps:
        if step['position_operands'] not in used_step_positions:
            # Skip steps that produce the final answer at the answer position
            # when we have a copy step (since the copy handles that)
            if copy_step_info and step['result'] == copy_step_info['result'] and \
               step['position_result'] == copy_step_info['position_result']:
                continue
            unused_steps_expr.append(format_step_expr(step))

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Forward Chaining - Sample {sample['sample_idx']}</title>
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
        .status.correct {{
            background-color: #2196F3;
            color: white;
            margin-left: 10px;
        }}
        .status.incorrect {{
            background-color: #ff9800;
            color: white;
            margin-left: 10px;
        }}
        .status.verified {{
            background-color: #4CAF50;
            color: white;
            margin-left: 10px;
        }}
        .status.partially-verified {{
            background-color: #ff9800;
            color: white;
            margin-left: 10px;
        }}
        .status.unverified {{
            background-color: #9e9e9e;
            color: white;
            margin-left: 10px;
        }}
        .question {{
            background-color: #e3f2fd;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
        }}
        .question-numbers {{
            background-color: #fce4ec;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
            font-family: monospace;
        }}
        .solution {{
            background-color: #e8f5e9;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
        }}
        .computation {{
            background-color: #fff3e0;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 14px;
        }}
        .unused-computation {{
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 13px;
            color: #666;
        }}
        .gt-comparison {{
            background-color: #fff;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .gt-comparison h3 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }}
        .gt-solution-box {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .gt-solution-item {{
            flex: 1;
            min-width: 300px;
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
        }}
        .gt-solution-item.primary {{
            border-color: #81c784;
            background-color: #f1f8e9;
        }}
        .gt-solution-item.best-fit {{
            border-color: #64b5f6;
            background-color: #e3f2fd;
        }}
        .gt-solution-item.same-tree {{
            border-color: #4caf50;
            border-width: 2px;
        }}
        .gt-solution-item h4 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #555;
        }}
        .gt-solution-item h4 .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 8px;
        }}
        .gt-solution-item h4 .badge.same {{
            background-color: #4caf50;
            color: white;
        }}
        .gt-solution-item h4 .badge.different {{
            background-color: #ff9800;
            color: white;
        }}
        .gt-steps {{
            font-family: monospace;
            font-size: 12px;
            background-color: rgba(255,255,255,0.7);
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .gt-stats {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .gt-stat {{
            font-size: 11px;
            padding: 4px 8px;
            background-color: rgba(255,255,255,0.8);
            border-radius: 3px;
            white-space: nowrap;
        }}
        .gt-stat strong {{
            color: #333;
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
        .cell-integer {{
            background-color: #fff;
        }}
        .cell-other {{
            background-color: #f9f9f9;
            color: #999;
        }}
        .cell-question-number {{
            background-color: #FFECB3;
            font-weight: bold;
            color: #333;
        }}
        .question-numbers-section {{
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }}
        .question-numbers-label {{
            font-weight: bold;
            font-size: 12px;
            color: #555;
            margin-right: 10px;
            white-space: nowrap;
        }}
        .question-numbers-cells {{
            display: flex;
            gap: 2px;
        }}
        .qnum-cell {{
            background-color: #FFECB3;
            border: 1px solid #FFD54F;
            padding: 4px 10px;
            font-family: monospace;
            font-size: 12px;
            font-weight: bold;
            text-align: center;
            min-width: 30px;
        }}
        /* Tree node highlighting (boxed) */
        .tree-node {{
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
        .steps-section {{
            margin: 12px 0;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }}
        .step-item {{
            display: inline-block;
            padding: 4px 10px;
            margin: 2px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
            color: white;
            vertical-align: top;
        }}
        .step-wrapper {{
            display: inline-block;
            text-align: center;
            margin: 2px 4px;
            vertical-align: top;
        }}
        .step-explainability {{
            font-size: 10px;
            margin-top: 2px;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        .step-explainability.explainable {{
            color: #2e7d32;
        }}
        .step-explainability.unverified {{
            color: #ef6c00;
        }}
        .step-explainability.unexplainable {{
            color: #c62828;
        }}
        .unused-step {{
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Forward Chaining Computation Tree (Top {top_k} tokens per position)</h1>
            <span class="status {'found' if tree_found else 'not-found'}">
                {'Computation Tree FOUND' if tree_found else 'No Computation Tree Found'}
            </span>
            <span class="status {'correct' if sample.get('answer_correct') else 'incorrect'}">
                {'Answer CORRECT' if sample.get('answer_correct') else 'Answer INCORRECT'}
            </span>
            {get_tree_status_badge(sample, metadata)}

            <div class="question">
                <strong>Question:</strong> {sample.get('question', 'N/A')}
            </div>

            <div class="question-numbers">
                <strong>Numbers in Question:</strong> {', '.join(str(n) for n in sample.get('question_numbers', [])) or 'None'}
            </div>

            <div class="solution">
                <strong>GT Solution:</strong> {sample.get('gt_solution', 'N/A')}
            </div>


            <div class="info-row">
                <div class="info-item"><strong>GT Answer:</strong> {sample.get('gt_answer', 'N/A')}</div>
                <div class="info-item"><strong>Model Answer:</strong> {sample.get('model_answer', 'N/A')}</div>
                <div class="info-item"><strong>Tree Steps:</strong> {best_tree['num_steps'] if best_tree else 0}</div>
                <div class="info-item"><strong>Total Steps Found:</strong> {sample.get('best_steps_found', 0)}</div>
                <div class="info-item"><strong>Unused Steps:</strong> {len(unused_steps_expr)}</div>
                <div class="info-item"><strong>Final Answer Type:</strong> {'Combined intermediates' if best_tree and best_tree.get('final_combination') else ('Direct step' if best_tree else 'N/A')}</div>
                <div class="info-item"><strong>Tree Explainable:</strong> {'Yes ✓' if sample.get('tree_explainable') else ('No ✗' if sample.get('tree_explainable') is False else 'N/A')}</div>
            </div>
"""

    # Add GT comparison info if available
    gt_comp = sample.get('gt_comparison')
    if gt_comp:
        primary = gt_comp.get('primary', {})
        best_fit = gt_comp.get('best_fit')

        html += """
            <div class="gt-comparison">
                <h3>Ground Truth Comparison</h3>
                <div class="gt-solution-box">
"""

        # Helper to render stats for a GT solution
        def render_gt_stats(comp):
            ir_count = comp.get('gt_intermediate_count', 'N/A')
            return f"""
                        <div class="gt-stats">
                            <span class="gt-stat"><strong>GT Steps:</strong> {comp.get('gt_step_count', 'N/A')}</span>
                            <span class="gt-stat"><strong>GT Intermediates:</strong> {ir_count}</span>
                            <span class="gt-stat"><strong>GT IRs in VP Top-1:</strong> {comp.get('gt_intermediates_in_vp_top1', 'N/A')}/{ir_count}</span>
                            <span class="gt-stat"><strong>GT IRs in used steps:</strong> {comp.get('gt_intermediates_in_used_steps', 'N/A')}/{ir_count}</span>
                            <span class="gt-stat"><strong>GT IRs in any found step:</strong> {comp.get('gt_intermediates_in_any_found', 'N/A')}/{ir_count}</span>
                            <span class="gt-stat"><strong>GT steps matching a used step:</strong> {comp.get('matching_used_steps', 'N/A')}/{comp.get('gt_step_count', 'N/A')}</span>
                            <span class="gt-stat"><strong>GT steps matching any found step:</strong> {comp.get('matching_any_steps', 'N/A')}/{comp.get('gt_step_count', 'N/A')}</span>
                        </div>"""

        # Primary GT solution
        primary_same = primary.get('is_same_tree', False)
        primary_badge = '<span class="badge same">SAME TREE</span>' if primary_same else '<span class="badge different">DIFFERENT</span>'
        primary_steps = primary.get('gt_steps', [])
        primary_steps_str = ' '.join(primary_steps) if primary_steps else sample.get('gt_solution', 'N/A')

        html += f"""
                    <div class="gt-solution-item primary {'same-tree' if primary_same else ''}">
                        <h4>Primary GT Solution {primary_badge}</h4>
                        <div class="gt-steps">{primary_steps_str}</div>
                        {render_gt_stats(primary)}
                    </div>
"""

        # Best fit GT solution (if different from primary)
        if best_fit:
            best_same = best_fit.get('is_same_tree', False)
            best_badge = '<span class="badge same">SAME TREE</span>' if best_same else '<span class="badge different">DIFFERENT</span>'
            best_steps = best_fit.get('gt_steps', [])
            best_steps_str = ' '.join(best_steps) if best_steps else 'N/A'
            best_idx = best_fit.get('solution_index', '?')

            html += f"""
                    <div class="gt-solution-item best-fit {'same-tree' if best_same else ''}">
                        <h4>Best Fit GT (gen_solutions[{best_idx}]) {best_badge}</h4>
                        <div class="gt-steps">{best_steps_str}</div>
                        {render_gt_stats(best_fit)}
                    </div>
"""

        html += """
                </div>
            </div>
"""

    # Build mapping from position_result to verification status from validation info
    verification_status = {}  # position_result -> bool (True = verified)
    validation_info = sample.get('validation')
    if validation_info and 'steps' in validation_info:
        for v_step in validation_info['steps']:
            verification_status[v_step['result_position']] = v_step.get('verified', False)

    # Add all steps display with colors
    if all_steps:
        # Build mapping from step (by position_operands) to tree step index for color matching
        tree_step_color_map = {}
        copy_step_for_display = best_tree.get('copy_step') if best_tree else None
        if best_tree:
            for tree_idx, tree_step in enumerate(best_tree.get('steps', [])):
                tree_step_color_map[tree_step['position_operands']] = tree_idx

        def render_step_item(step, show_used_label=True):
            """Render a single step item with proper formatting."""
            used = step.get('used_in_tree', False)
            # Use tree step index for color if this step is used in tree, otherwise use grey
            if used and step['position_operands'] in tree_step_color_map:
                color_idx = tree_step_color_map[step['position_operands']]
                color = STEP_COLORS[color_idx % len(STEP_COLORS)]
            else:
                color = "#999999"  # Grey for unused steps

            # Check explainability (pre-computed in run.py for all steps)
            is_explainable = step.get('explainable', False)

            # Check verification status
            pos_result = step['position_result']
            is_verified = verification_status.get(pos_result, None)

            used_class = '' if used else 'unused-step'
            op = step['operation']
            if step.get('operand3') is not None:
                expr_body = format_3op_expression(step['operand1'], step['operand2'], step['operand3'], op, sep=" ")
                expr = f"{expr_body} = {step['result']}"
            else:
                expr = f"{step['operand1']} {op} {step['operand2']} = {step['result']}"

            # Build status label
            if is_explainable:
                if is_verified is True:
                    status_text = 'Explainable, Verified'
                    status_css = 'explainable'
                elif is_verified is False:
                    status_text = 'Explainable, Unverified'
                    status_css = 'unverified'
                else:
                    status_text = 'Explainable'
                    status_css = 'explainable'
            else:
                # Unexplainable - but may still have verification result
                if is_verified is True:
                    status_text = 'Unexplainable, Verified'
                    status_css = 'unexplainable'
                elif is_verified is False:
                    status_text = 'Unexplainable, Unverified'
                    status_css = 'unexplainable'
                else:
                    status_text = 'Unexplainable'
                    status_css = 'unexplainable'

            # Add used/unused suffix if requested
            if show_used_label:
                status_text += ', Used' if used else ', Unused'

            result = f'                <div class="step-wrapper">\n'
            result += f'                    <span class="step-item {used_class}" style="background-color: {color};">'
            result += f'[→{pos_result}] {expr}</span>\n'
            result += f'                    <div class="step-explainability {status_css}">{status_text}</div>\n'
            result += f'                </div>\n'
            return result

        # Discovered Computation section (used steps only)
        html += '            <div class="steps-section">\n'
        html += '                <strong>Discovered Computation:</strong><br>\n'
        if best_tree and best_tree.get('steps'):
            for tree_step in best_tree['steps']:
                # Find the corresponding step in all_steps
                matching_step = next((s for s in all_steps if s['position_operands'] == tree_step['position_operands']), None)
                if matching_step:
                    html += render_step_item(matching_step, show_used_label=False)
                else:
                    # Fallback: render tree_step directly
                    tree_step_copy = dict(tree_step)
                    tree_step_copy['used_in_tree'] = True
                    tree_step_copy['explainable'] = tree_step.get('explainable', True)
                    html += render_step_item(tree_step_copy, show_used_label=False)

            # Also render final combination step if present (e.g., "9 + 9 = 18")
            final_combo = best_tree.get('final_combination')
            if final_combo:
                values = final_combo.get('values', [])
                op = final_combo.get('operation', '?')
                final_answer = best_tree.get('final_answer')
                # Use the color for the final step (after all tree steps)
                combo_step_idx = len(best_tree['steps'])
                combo_color = STEP_COLORS[combo_step_idx % len(STEP_COLORS)]

                # Format the expression
                if len(values) == 2:
                    expr = f"{values[0]} {op} {values[1]} = {final_answer}"
                elif len(values) == 3:
                    expr = format_3op_expression(values[0], values[1], values[2], op, sep=" ")
                    expr = f"{expr} = {final_answer}"
                else:
                    expr = f"{' '.join(map(str, values))} = {final_answer}"

                html += f'                <div class="step-wrapper">\n'
                html += f'                    <span class="step-item" style="background-color: {combo_color};">'
                html += f'[→answer] {expr}</span>\n'
                html += f'                    <div class="step-explainability explainable">Explainable, Verified</div>\n'
                html += f'                </div>\n'
        else:
            html += '                <span style="color: #666;">No computation tree found</span>\n'
        html += '            </div>\n'

        # All Discovered Steps section (all steps, including unused)
        html += '            <div class="steps-section">\n'
        html += '                <strong>All Discovered Steps:</strong><br>\n'
        for i, step in enumerate(all_steps):
            # Skip steps that produce the final answer at the answer position
            # when we have a copy step (since the copy handles that)
            if copy_step_for_display and step['result'] == copy_step_for_display['result'] and \
               step['position_result'] == copy_step_for_display['position_result']:
                continue

            html += render_step_item(step, show_used_label=True)
        html += '            </div>\n'

    html += """        </div>

        <div class="table-container">
            <div class="table-wrapper" id="tableWrapper">
"""

    # Add question numbers section ABOVE the table if include_question_tokens was used
    if include_question_tokens and question_numbers:
        html += """                <div class="question-numbers-section">
                    <span class="question-numbers-label">Numbers from the question:</span>
                    <div class="question-numbers-cells">
"""
        for qnum in question_numbers:
            html += f'                        <div id="qnum_{qnum}" class="qnum-cell">{qnum}</div>\n'
        html += """                    </div>
                </div>
"""

    html += """                <table id="vocabTable">
                    <thead>
                        <tr>
                            <th>Rank</th>
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

                css_class = f"cell-{cell_type}"

                # Add tree-node class if this cell is in the best tree
                if cell_id in tree_nodes:
                    css_class += " tree-node"

                html += f'                            <td id="{cell_id}" class="{css_class}">{token_display}</td>\n'
            else:
                html += f'                            <td class="cell-other">-</td>\n'

        html += f"                        </tr>\n"

    # Add bottom row with input tokens
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
                    <div class="legend-color boxed" style="background-color: #A5D6A7;"></div>
                    <span>Intermediate Result</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color boxed" style="background-color: #90CAF9;"></div>
                    <span>Final Answer</span>
                </div>
"""

    # Add question number legend if include_question_tokens was used
    if include_question_tokens:
        html += """                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFECB3;"></div>
                    <span>Question Number</span>
                </div>
"""

    # Add step color legend with arrows
    if best_tree and best_tree.get('steps'):
        for i, step in enumerate(best_tree['steps']):
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
                    <span>Step {i + 1}: {format_legend_step(step)}</span>
                </div>
"""
        # Add copy step if present
        copy_step = best_tree.get('copy_step')
        if copy_step:
            copy_step_idx = len(best_tree['steps'])
            copy_color = STEP_COLORS[copy_step_idx % len(STEP_COLORS)]
            copy_val = copy_step['operand1']
            html += f"""                <div class="legend-item">
                    <svg width="30" height="20" style="vertical-align: middle;">
                        <defs>
                            <marker id="legend-arrow-copy" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="{copy_color}" />
                            </marker>
                        </defs>
                        <line x1="2" y1="10" x2="22" y2="10" stroke="{copy_color}" stroke-width="2" marker-end="url(#legend-arrow-copy)" />
                    </svg>
                    <span>Copy: {copy_val} \u2192 ans</span>
                </div>
"""

        # Add final combination step if present
        final_combo = best_tree.get('final_combination')
        if final_combo:
            values = final_combo.get('values', [])
            op = final_combo.get('operation', '?')
            v1 = values[0] if len(values) > 0 else '?'
            v2 = values[1] if len(values) > 1 else '?'
            final_answer = best_tree.get('final_answer')
            combo_step_idx = len(best_tree['steps'])
            combo_color = STEP_COLORS[combo_step_idx % len(STEP_COLORS)]
            html += f"""                <div class="legend-item">
                    <svg width="30" height="20" style="vertical-align: middle;">
                        <defs>
                            <marker id="legend-arrow-combo" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="{combo_color}" />
                            </marker>
                        </defs>
                        <line x1="2" y1="10" x2="22" y2="10" stroke="{combo_color}" stroke-width="2" marker-end="url(#legend-arrow-combo)" />
                    </svg>
                    <span>Final: {v1} {op} {v2} = {final_answer}</span>
                </div>
"""

    # Add copy from question legend if any
    if copy_from_question_steps:
        html += """                <div class="legend-item">
                    <svg width="30" height="20" style="vertical-align: middle;">
                        <defs>
                            <marker id="legend-arrow-copy-q" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#FF9800" />
                            </marker>
                        </defs>
                        <line x1="2" y1="10" x2="22" y2="10" stroke="#FF9800" stroke-width="2" marker-end="url(#legend-arrow-copy-q)" />
                    </svg>
                    <span>Copy from Question</span>
                </div>
"""

    # Add unused steps legend if any
    num_unused = sample.get('num_unused_steps', 0)
    if num_unused > 0:
        html += """                <div class="legend-item">
                    <svg width="30" height="20" style="vertical-align: middle;">
                        <defs>
                            <marker id="legend-arrow-unused" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                <polygon points="0 0, 10 3.5, 0 7" fill="#999999" />
                            </marker>
                        </defs>
                        <line x1="2" y1="10" x2="22" y2="10" stroke="#999999" stroke-width="1.5" stroke-dasharray="4,2" marker-end="url(#legend-arrow-unused)" />
                    </svg>
                    <span>Unused Step</span>
                </div>
"""

    html += """            </div>
        </div>
    </div>

    <script>
"""

    # Add edges data for JavaScript
    html += f"        const edgesData = {json.dumps(edges)};\n"

    html += """
        function drawLines() {
            const svg = document.getElementById('svgOverlay');
            const wrapper = document.getElementById('tableWrapper');
            const table = document.getElementById('vocabTable');

            // Size SVG to wrapper (includes question numbers section above table)
            svg.style.width = wrapper.offsetWidth + 'px';
            svg.style.height = wrapper.offsetHeight + 'px';
            svg.setAttribute('width', wrapper.offsetWidth);
            svg.setAttribute('height', wrapper.offsetHeight);

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

            // Group edges by (from_id, to_id) to offset overlapping arrows
            const edgeGroups = {};
            edgesData.forEach((edge, idx) => {
                const key = edge.from_id + '->' + edge.to_id;
                if (!edgeGroups[key]) edgeGroups[key] = [];
                edgeGroups[key].push({...edge, idx: idx});
            });

            // Group edges by target to offset arrows ending at same cell
            const targetGroups = {};
            edgesData.forEach((edge, idx) => {
                const key = edge.to_id;
                if (!targetGroups[key]) targetGroups[key] = [];
                targetGroups[key].push({...edge, idx: idx});
            });

            edgesData.forEach((edge, edgeIdx) => {
                const fromCell = document.getElementById(edge.from_id);
                const toCell = document.getElementById(edge.to_id);

                if (fromCell && toCell) {
                    const fromRect = fromCell.getBoundingClientRect();
                    const toRect = toCell.getBoundingClientRect();
                    const wrapperRect = wrapper.getBoundingClientRect();

                    // Get cell centers
                    let fromX = fromRect.left + fromRect.width / 2 - wrapperRect.left;
                    let fromY = fromRect.top + fromRect.height / 2 - wrapperRect.top;
                    let toX = toRect.left + toRect.width / 2 - wrapperRect.left;
                    let toY = toRect.top + toRect.height / 2 - wrapperRect.top;

                    // Offset start/end points away from center (like matplotlib impl)
                    const offsetX = fromRect.width * 0.18;
                    const offsetY = fromRect.height * 0.18;

                    // Start point: nudge toward target
                    if (toX > fromX) fromX += offsetX;
                    else if (toX < fromX) fromX -= offsetX;
                    if (toY > fromY) fromY += offsetY;
                    else if (toY < fromY) fromY -= offsetY;

                    // End point: nudge toward source
                    if (fromX < toX) toX -= offsetX;
                    else if (fromX > toX) toX += offsetX;
                    if (fromY < toY) toY -= offsetY;
                    else if (fromY > toY) toY += offsetY;

                    // Calculate arc radius like matplotlib's arc3
                    // rad controls curve perpendicular to the line
                    const dx = toX - fromX;
                    const dy = toY - fromY;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    // Scale rad with distance, cap it (matches matplotlib impl)
                    let rad = Math.min(0.15 + 0.04 * Math.abs(dx), 0.22);

                    // Offset for multiple arrows to same target
                    const targetGroup = targetGroups[edge.to_id];
                    const groupIdx = targetGroup.findIndex(e => e.idx === edgeIdx);
                    const groupSize = targetGroup.length;
                    rad += groupIdx * 0.08;

                    // Control point perpendicular to the line (arc3 style)
                    const midX = (fromX + toX) / 2;
                    const midY = (fromY + toY) / 2;

                    // Perpendicular direction (rotate 90 degrees, normalize)
                    const perpX = -dy / dist;
                    const perpY = dx / dist;

                    // Control point offset perpendicular to line
                    const ctrlOffset = dist * rad;
                    const ctrlX = midX + perpX * ctrlOffset;
                    const ctrlY = midY + perpY * ctrlOffset;

                    // Create quadratic Bezier curve path
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    const d = `M ${fromX} ${fromY} Q ${ctrlX} ${ctrlY} ${toX} ${toY}`;
                    path.setAttribute('d', d);
                    path.setAttribute('stroke', edge.color);
                    path.setAttribute('stroke-width', edge.used ? '2' : '1.5');
                    path.setAttribute('fill', 'none');
                    if (!edge.used) {
                        path.setAttribute('stroke-dasharray', '4,2');
                    }
                    const colorId = edge.color.replace('#', '');
                    path.setAttribute('marker-end', 'url(#arrowhead-' + colorId + ')');

                    svg.appendChild(path);
                }
            });
        }

        window.addEventListener('load', drawLines);
        window.addEventListener('resize', drawLines);
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
    tree_found_samples: List[int],
    no_tree_samples: List[int],
    metadata: Dict = None
):
    """Create an index.html linking to all visualizations."""

    # Build validation settings string for header
    validation_settings_html = ""
    if metadata and metadata.get('validate'):
        validation_n = metadata.get('validation_n', 3)
        required_passes = metadata.get('validation_required_passes', 2)
        max_rank = metadata.get('validation_max_rank', 1)
        validation_settings_html = f"""
        <div class="validation-settings">
            <strong>Validation Settings:</strong> {required_passes}/{validation_n} passes required, top-{max_rank} rank
        </div>
"""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Forward Chaining Visualizations - Index</title>
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
        .sample-link .info {
            color: #666;
            font-size: 14px;
        }
        .sample-link .correct {
            color: #4CAF50;
            font-weight: bold;
        }
        .sample-link .incorrect {
            color: #f44336;
            font-weight: bold;
        }
        .validation-settings {
            background-color: #e3f2fd;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Forward Chaining Visualizations</h1>
""" + validation_settings_html + """
        <div class="section found">
            <h2>Computation Tree Found</h2>
"""

    for sample_idx in tree_found_samples:
        sample = next((s for s in results if s['sample_idx'] == sample_idx), None)
        if sample:
            correct_class = 'correct' if sample.get('answer_correct') else 'incorrect'
            correct_text = '✓' if sample.get('answer_correct') else '✗'
            num_steps = sample.get('best_tree', {}).get('num_steps', 0) if sample.get('best_tree') else 0
            html += f"""            <a href="sample_{sample_idx:03d}_tree_found.html" class="sample-link">
                <span class="idx">Sample {sample_idx}</span>
                <span class="info">GT: {sample.get('gt_answer', 'N/A')} | Model: {sample.get('model_answer', 'N/A')} | Steps: {num_steps}</span>
                <span class="{correct_class}">{correct_text}</span>
            </a>
"""

    html += """        </div>

        <div class="section not-found">
            <h2>No Computation Tree Found</h2>
"""

    for sample_idx in no_tree_samples:
        sample = next((s for s in results if s['sample_idx'] == sample_idx), None)
        if sample:
            correct_class = 'correct' if sample.get('answer_correct') else 'incorrect'
            correct_text = '✓' if sample.get('answer_correct') else '✗'
            html += f"""            <a href="sample_{sample_idx:03d}_no_tree.html" class="sample-link">
                <span class="idx">Sample {sample_idx}</span>
                <span class="info">GT: {sample.get('gt_answer', 'N/A')} | Model: {sample.get('model_answer', 'N/A')}</span>
                <span class="{correct_class}">{correct_text}</span>
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
        description='Create HTML visualizations of forward chaining computation trees'
    )

    parser.add_argument(
        '--results_json',
        type=str,
        required=True,
        help='Path to results.json from forward chaining run.py'
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
        default=10,
        help='Number of "tree found" examples to visualize'
    )

    parser.add_argument(
        '--num_not_found',
        type=int,
        default=5,
        help='Number of "no tree" examples to visualize'
    )

    parser.add_argument(
        '--sample_indices',
        type=int,
        nargs='+',
        default=None,
        help='Specific sample indices to visualize (overrides num_found/num_not_found)'
    )

    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to visualize (all samples up to this limit)'
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
    model_type = metadata.get('model_type', 'coconut')
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
        all_samples_to_viz = args.sample_indices
        tree_found_samples = [idx for idx in all_samples_to_viz
                              if any(s['sample_idx'] == idx and s.get('tree_found') for s in per_sample)]
        no_tree_samples = [idx for idx in all_samples_to_viz if idx not in tree_found_samples]
    elif args.max_samples is not None:
        # Visualize all samples up to max_samples
        all_sample_indices = [s['sample_idx'] for s in per_sample[:args.max_samples]]
        tree_found_samples = [s['sample_idx'] for s in per_sample[:args.max_samples] if s.get('tree_found')]
        no_tree_samples = [s['sample_idx'] for s in per_sample[:args.max_samples] if not s.get('tree_found')]
        all_samples_to_viz = all_sample_indices
    else:
        tree_found_samples = []
        no_tree_samples = []

        for sample in per_sample:
            if sample.get('tree_found'):
                tree_found_samples.append(sample['sample_idx'])
            else:
                no_tree_samples.append(sample['sample_idx'])

        tree_found_samples = tree_found_samples[:args.num_found]
        no_tree_samples = no_tree_samples[:args.num_not_found]
        all_samples_to_viz = tree_found_samples + no_tree_samples

    logging.info(f"Tree found samples to visualize: {tree_found_samples}")
    logging.info(f"No tree samples to visualize: {no_tree_samples}")
    success_count = 0

    for sample_idx in all_samples_to_viz:
        sample = next((s for s in per_sample if s['sample_idx'] == sample_idx), None)
        if sample is None:
            logging.warning(f"Sample {sample_idx} not found in results")
            continue

        # Determine filename
        tree_status = "tree_found" if sample.get('tree_found') else "no_tree"
        output_path = output_dir / f"sample_{sample_idx:03d}_{tree_status}.html"

        if create_visualization_html(sample, output_path, top_k=top_k, model_type=model_type, num_latent=num_latent, metadata=metadata):
            logging.info(f"Created visualization for sample {sample_idx}")
            success_count += 1
        else:
            logging.warning(f"Failed to create visualization for sample {sample_idx}")

    # Create index
    create_index_html(per_sample, output_dir, tree_found_samples, no_tree_samples, metadata=metadata)

    logging.info(f"\nCreated {success_count} visualizations in {output_dir}")
    logging.info(f"Open {output_dir / 'index.html'} to browse")


if __name__ == "__main__":
    main()
