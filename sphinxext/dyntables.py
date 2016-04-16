r""" Directive for dynamic generation of tables from code

Example::

    .. dynamic-math-table::  **Title of table**
        :header: "", "First", "Second"
        :raw-cols: 0

        # raw-cols above specifies columns that don't have math symbols
        import sympy
        a, b, c, d = sympy.symbols(r'\alpha, \beta, \gamma, \delta')
        [["one", a, b], ["two", c, d]]
"""

from docutils import nodes, statemachine
from docutils.parsers.rst.directives.tables import CSVTable
from docutils.parsers.rst import directives

from docutils.utils import SystemMessagePropagation

from texext.mathcode import eval_code

def nonnegative_int_list(argument):
    """
    Converts a space- or comma-separated list of values into a Python list
    of integers.

    (Directive option conversion function.)

    Raises ValueError for values that aren't non-negative integers.
    """
    if ',' in argument:
        entries = argument.split(',')
    else:
        entries = argument.split()
    return [directives.nonnegative_int(entry) for entry in entries]


class DynamicTable(CSVTable):
    option_spec = {'header-rows': directives.nonnegative_int,
                   'stub-columns': directives.nonnegative_int,
                   'header': directives.unchanged,
                   'widths': directives.positive_int_list,
                   'file': directives.path,
                   'url': directives.uri,
                   'encoding': directives.encoding,
                   'class': directives.class_option,
                   'name': directives.unchanged,
                   # field delimiter char
                   'delim': directives.single_char_or_whitespace_or_unicode,
                   # treat whitespace after delimiter as significant
                   'keepspace': directives.flag,
                   # text field quote/unquote char:
                   'quote': directives.single_char_or_unicode,
                   # char used to escape delim & quote as-needed:
                   'escape': directives.single_char_or_unicode,
                   'newcontext': directives.flag,
                  }

    def get_plot_context(self):
        # First try dyntable_plot_context dictionary
        plot_context = setup.config.dyntable_plot_context
        if plot_context is not None:
            # Plot context is a string naming a module attribute
            parts = plot_context.split('.')
            mod_name, el_name = '.'.join(parts[:-1]), parts[-1]
            mod = __import__(mod_name, globals(), locals(), el_name)
            return getattr(mod, el_name)
        # Default to matplotlib plot_context dictionary
        from matplotlib.sphinxext.plot_directive import plot_context
        return plot_context

    def get_context(self, newcontext=False):
        if setup.config.dyntable_use_plot_ns:
            plot_context = self.get_plot_context()
        else:
            plot_context = setup.dyntable_code_context
        if newcontext:
            plot_context.clear()
        return plot_context

    def run(self):
        self.check_requirements()
        title, messages = self.make_title()
        table_head, max_header_cols = self.process_header_option()
        rows, source = self.get_rows()
        max_cols = max(len(row) for row in rows)
        max_cols = max(max_cols, max_header_cols)
        header_rows = self.options.get('header-rows', 0)
        stub_columns = self.options.get('stub-columns', 0)
        self.check_table_dimensions(rows, header_rows, stub_columns)
        table_head.extend(rows[:header_rows])
        table_body = rows[header_rows:]
        col_widths = self.get_column_widths(max_cols)
        self.extend_short_rows_with_empty_cells(max_cols,
                                                (table_head, table_body))
        table = (col_widths, table_head, table_body)
        table_node = self.state.build_table(table, self.content_offset,
                                            stub_columns)
        table_node['classes'] += self.options.get('class', [])
        self.add_name(table_node)
        if title:
            table_node.insert(0, title)
        return [table_node] + messages

    def get_rows(self):
        """
        Get rows as list of lists or array from the directive content
        """
        if not self.content:
            error = self.state_machine.reporter.warning(
                'The "%s" directive requires content; none supplied.'
                % self.name, nodes.literal_block(
                self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        want_new = True if 'newcontext' in self.options else False
        context = self.get_context(want_new)
        source = self.content.source(0)
        output = eval_code('\n'.join(self.content), context)
        rows = self._process_output(output)
        rows = [] if rows is None else rows
        return self._process_rows(rows, source), source

    def _process_rows(self, rows, source):
        """ Add table cell boilerplace to cells in rows
        """
        p_rows = []
        for row in rows:
            p_row = []
            for cell in row:
                cell_content = statemachine.StringList(
                    cell.splitlines(),
                    source=source)
                p_row.append((0, 0, 0, cell_content))
            p_rows.append(p_row)
        return p_rows

    def _process_output(self, output):
        """ Apply any post-processing to output of code
        """
        return output


class DynamicMathTable(DynamicTable):
    option_spec = DynamicTable.option_spec.copy()
    option_spec['raw-cols'] = nonnegative_int_list

    def _process_output(self, output):
        """ Apply sympy.latex and add math role to selected columns
        """
        raw_cols = self.options.get('raw-cols', [])
        from sympy import latex
        rows = []
        for row in output:
            cells = []
            for col_no, cell in enumerate(row):
                if col_no not in raw_cols:
                    cell = ':math:`{}`'.format(latex(cell))
                cells.append(cell)
            rows.append(cells)
        return rows


def setup(app):
    # Global variables
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    # Workspace for code run in dyntable blocks
    setup.dyntable_code_context = dict()
    app.add_directive('dynamic-table', DynamicTable)
    app.add_directive('dynamic-math-table', DynamicMathTable)
    app.add_config_value('dyntable_use_plot_ns', False, 'env')
    app.add_config_value('dyntable_plot_context', None, 'env')
