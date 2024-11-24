from nicegui import ui


class PromptTemplateVariablesComponent:
    def __init__(self, variable_names: list[str]):
        self.variable_names = variable_names
        self.variable_container: ui.element | None = None
        self.render()

    def collect_data(self) -> list[dict[str, str]]:
        data = []
        for element in self.variable_container.descendants():
            if isinstance(element, ui.textarea):
                if element.props.get('label') == self.variable_names[0]:
                    data.append({})
                data[-1][element.props.get('label')] = element.props.get('value')
        return data

    def add_variables(self):
        with self.variable_container:
            with ui.row().classes('items-center gap-2') as row:
                with ui.column():
                    for variable_name in self.variable_names:
                        ui.textarea(label=f"{variable_name}").props('outlined')
                with ui.column():
                    ui.button(icon='remove', on_click=lambda: row.delete()).props('size=sm round')

    def render(self):
        with ui.card():
            with ui.card_section():
                ui.label('Prompt Variables').classes('text-h6')
            self.variable_container = ui.element('div').classes('w-full')
            self.add_variables()
            with ui.row().classes('w-full justify-center'):
                ui.button(icon='add', on_click=self.add_variables)

            with ui.card_actions().classes('w-full justify-end'):
                ui.button(text='Cancel')
                ui.button(text='Save')
