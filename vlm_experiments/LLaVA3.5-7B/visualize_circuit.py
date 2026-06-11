import json
import argparse
import os

def generate_svg(json_path, output_path):
    print(f"Reading circuit from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    circuit = data["circuit"]
    
    # Constants for grid spacing
    # Feel free to adjust these spacing parameters to make the grid larger/smaller
    layer_width = 120
    head_height = 50
    padding_x = 80
    padding_y = 80
    
    # Assuming LLaVA 7B structure (32 layers, 32 heads)
    max_layer = 32
    max_head = 32
    
    width = max_layer * layer_width + padding_x * 2 + 200 # extra space for logits on the right
    height = max_head * head_height + padding_y * 2
    
    svg = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    .node-text { font-family: "Inter", "Arial", sans-serif; font-size: 14px; font-weight: bold; pointer-events: none; }',
        '    .score-text { font-family: "Inter", "Arial", sans-serif; font-size: 11px; fill: #444; pointer-events: none; }',
        '    .axis-text { font-family: "Inter", "Arial", sans-serif; font-size: 24px; font-weight: bold; fill: #666; }',
        '    .label-text { font-family: "Inter", "Arial", sans-serif; font-size: 16px; font-weight: bold; fill: #888; }',
        '    .grid-line { stroke: #eee; stroke-width: 1; }',
        '  </style>',
        '  <defs>',
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '      <path d="M0,0 L0,6 L9,3 z" fill="#888" />',
        '    </marker>',
        '  </defs>',
        # Background
        f'  <rect width="100%" height="100%" fill="#fafafa"/>'
    ]
    
    # Draw Background Grid & Axes Labels
    svg.append(f'  <text x="{width/2}" y="{padding_y - 30}" class="axis-text" text-anchor="middle">Network Layers (0 to 31) &#8594;</text>')
    svg.append(f'  <text x="{padding_x - 40}" y="{height/2}" class="axis-text" text-anchor="middle" transform="rotate(-90 {padding_x - 40},{height/2})">Attention Heads (0 to 31) &#8594;</text>')

    for l in range(max_layer):
        x = l * layer_width + padding_x
        svg.append(f'  <line x1="{x}" y1="{padding_y}" x2="{x}" y2="{height - padding_y}" class="grid-line" />')
        svg.append(f'  <text x="{x}" y="{padding_y - 10}" class="label-text" text-anchor="middle">L{l}</text>')

    for h in range(max_head):
        y = h * head_height + padding_y
        svg.append(f'  <line x1="{padding_x}" y1="{y}" x2="{max_layer * layer_width + padding_x}" y2="{y}" class="grid-line" />')
        svg.append(f'  <text x="{padding_x - 10}" y="{y+5}" class="label-text" text-anchor="end">H{h}</text>')

    # Store computed positions for all nodes
    positions = {}
    
    # Setup Logits Node
    logits_x = max_layer * layer_width + padding_x + 100
    logits_y = ((max_head - 1) / 2) * head_height + padding_y
    positions["logits"] = (logits_x, logits_y)
    
    # Setup Circuit Nodes
    for node in circuit:
        l = node["layer"]
        h = node["head"]
        name = f"L{l}H{h}"
        
        x = l * layer_width + padding_x
        y = h * head_height + padding_y
        positions[name] = (x, y)
        
    # Phase 1: Draw ALL Edges (so they render behind nodes)
    for node in circuit:
        name = f"L{node['layer']}H{node['head']}"
        x1, y1 = positions[name]
        
        targets = [t.strip() for t in node["target"].split(",")]
        
        for t in targets:
            if t in positions:
                x2, y2 = positions[t]
                # A little math to make the arrow hit the edge of the circle (radius ~30) instead of the center
                import math
                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy)
                if dist == 0: continue
                
                # offset start and end by radius
                offset_start = 30
                offset_end = 35 if t != "logits" else 45
                
                start_x = x1 + (dx/dist) * offset_start
                start_y = y1 + (dy/dist) * offset_start
                end_x = x2 - (dx/dist) * offset_end
                end_y = y2 - (dy/dist) * offset_end
                
                svg.append(f'  <line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" stroke="#888" stroke-width="2.5" marker-end="url(#arrow)" />')

    # Phase 2: Draw Logits Node
    svg.append(f'  <rect x="{logits_x-40}" y="{logits_y-25}" width="80" height="50" fill="#ccffcc" stroke="#33cc33" stroke-width="3" rx="8"/>')
    svg.append(f'  <text x="{logits_x}" y="{logits_y+5}" class="node-text" text-anchor="middle">Logits</text>')

    # Phase 3: Draw Circuit Nodes
    for node in circuit:
        name = f"L{node['layer']}H{node['head']}"
        score = node["combined_score"]
        label = node["label"]
        
        x, y = positions[name]
        
        # Color coding: Light Red for promoter, Light Blue for suppressor
        fill_color = "#ffcccc" if "promoter" in label else "#ccccff"
        stroke_color = "#cc0000" if "promoter" in label else "#0000cc"
        
        svg.append(f'  <circle cx="{x}" cy="{y}" r="30" fill="{fill_color}" stroke="{stroke_color}" stroke-width="3" />')
        svg.append(f'  <text x="{x}" y="{y-2}" class="node-text" text-anchor="middle">{name}</text>')
        svg.append(f'  <text x="{x}" y="{y+15}" class="score-text" text-anchor="middle">{score:.3f}</text>')
        
    svg.append('</svg>')
    
    # Save the SVG
    with open(output_path, 'w') as f:
        f.write("\n".join(svg))
        
    print(f"High-dimensional SVG visualization saved successfully to {output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a layer-vs-head SVG visualization of the circuit")
    parser.add_argument("--input", type=str, default="circuit_result.json", help="Path to JSON output from find_circuit_from_matrix.py")
    parser.add_argument("--output", type=str, default="circuit_visualization.svg", help="Output SVG filename")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        generate_svg(args.input, args.output)
    else:
        print(f"Error: Could not find {args.input}. Please run find_circuit_from_matrix.py first to generate the JSON.")
