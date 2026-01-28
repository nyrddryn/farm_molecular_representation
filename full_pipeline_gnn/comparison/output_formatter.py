"""
Output Formatting

This module handles formatting of comparison results,
converting graph structures to human-readable output.
"""


class OutputFormatter:
    """
    Format comparison results for display
    """

    def extract_atom_groups(self, graph, node_indices):
        """
        Extract atom-level representation from FG node indices

        Args:
            graph (dict): FG graph structure
            node_indices (list): List of node indices

        Returns:
            list: List of base atoms (without FG labels)
        """
        atoms = []
        for node_idx in node_indices:
            fg_token = graph['nodes'][node_idx]
            # Get base atom (before first underscore)
            if '_' in fg_token:
                base_atom = fg_token.split('_')[0]
                atoms.append(base_atom)
            else:
                atoms.append(fg_token)
        return atoms

    def format_diff_group_atoms(self, graph, node_indices):
        """
        Format difference group at atom-level (not FG-level)

        Args:
            graph (dict): FG graph structure
            node_indices (list): List of node indices in group

        Returns:
            str: Formatted atom group string
        """
        atoms = self.extract_atom_groups(graph, node_indices)

        if len(atoms) == 0:
            return ""
        elif len(atoms) == 1:
            return atoms[0]
        else:
            # Group consecutive atoms
            return f"({''.join(atoms)})"

    def format_diff_group_with_fg(self, graph, node_indices):
        """
        Format difference group with FG labels

        Args:
            graph (dict): FG graph structure
            node_indices (list): List of node indices in group

        Returns:
            str: Formatted FG group string
        """
        fg_tokens = [graph['nodes'][i] for i in node_indices]

        if len(fg_tokens) == 0:
            return ""
        elif len(fg_tokens) == 1:
            return fg_tokens[0]
        else:
            # Join with underscore
            return f"({'_'.join(fg_tokens)})"

    def format_comparison_result(self, graph1, graph2,
                                 matched_nodes1, matched_nodes2,
                                 diff_groups1, diff_groups2,
                                 use_fg_labels=False,
                                 edge_analysis=None):
        """
        Format the complete comparison result

        Args:
            graph1 (dict): First FG graph
            graph2 (dict): Second FG graph
            matched_nodes1 (list): Matched node indices in graph1
            matched_nodes2 (list): Matched node indices in graph2
            diff_groups1 (list): Difference groups in graph1
            diff_groups2 (list): Difference groups in graph2
            use_fg_labels (bool): Use FG labels instead of atoms
            edge_analysis (dict): Edge verification results

        Returns:
            str: Formatted comparison result
        """
        # Build MCS from matched nodes (atom-level, no FG labels)
        if matched_nodes1:
            # Extract base atoms from matched nodes
            mcs_atoms = []
            for node_idx in matched_nodes1:
                fg_token = graph1['nodes'][node_idx]
                base_atom = fg_token.split('_')[0] if '_' in fg_token else fg_token
                mcs_atoms.append(base_atom)

            # Simple concatenation - no RDKit validation
            mcs_string = ''.join(mcs_atoms)
        else:
            mcs_string = ""

        # Format diff groups (choose format based on flag)
        serial_parts1 = []
        for i, group in enumerate(diff_groups1, 1):
            if use_fg_labels:
                group_label = self.format_diff_group_with_fg(graph1, group)
            else:
                group_label = self.format_diff_group_atoms(graph1, group)

            if group_label:
                # Get position for ordering
                min_pos = min(graph1['node_indices'][node_idx] for node_idx in group)
                serial_parts1.append((min_pos, group_label, i))

        serial_parts2 = []
        for i, group in enumerate(diff_groups2, 1):
            if use_fg_labels:
                group_label = self.format_diff_group_with_fg(graph2, group)
            else:
                group_label = self.format_diff_group_atoms(graph2, group)

            if group_label:
                min_pos = min(graph2['node_indices'][node_idx] for node_idx in group)
                serial_parts2.append((min_pos, group_label, i))

        # Sort by position
        serial_parts1.sort(key=lambda x: x[0])
        serial_parts2.sort(key=lambda x: x[0])

        # Build output with proper positioning
        # Determine MCS position based on first matched node
        if matched_nodes1:
            mcs_pos1 = min(graph1['node_indices'][i] for i in matched_nodes1)
        else:
            mcs_pos1 = 0

        if matched_nodes2:
            mcs_pos2 = min(graph2['node_indices'][i] for i in matched_nodes2)
        else:
            mcs_pos2 = 0

        # Build mol1 output - insert diff groups and MCS at proper positions
        all_parts1 = []
        for pos, fg, serial in serial_parts1:
            all_parts1.append((pos, f"{fg}:{serial}"))
        all_parts1.append((mcs_pos1, mcs_string))
        all_parts1.sort(key=lambda x: x[0])

        output1 = " ".join([part[1] for part in all_parts1])

        # Build mol2 output
        all_parts2 = []
        for pos, fg, serial in serial_parts2:
            all_parts2.append((pos, f"{fg}:{serial}"))
        all_parts2.append((mcs_pos2, mcs_string))
        all_parts2.sort(key=lambda x: x[0])

        output2 = " ".join([part[1] for part in all_parts2])

        result = f"{output1} and {output2}"

        # Add edge topology warning if mismatch detected
        if edge_analysis and edge_analysis['topology1'] != edge_analysis['topology2']:
            result += f" [⚠️ TOPOLOGY: {edge_analysis['topology1']} vs {edge_analysis['topology2']}]"

        if edge_analysis and edge_analysis['edge_consistency'] < 1.0:
            result += f" [Edge consistency: {edge_analysis['edge_consistency']:.2%}]"

        return result
