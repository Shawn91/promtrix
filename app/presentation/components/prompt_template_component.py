from nicegui import ui

from app.domain.entities import PromptTemplate
from app.presentation.base import Component


class PromptTemplatesComponent(Component):
    def __init__(self):
        self.card = ui.card()
        self.inputs_container: ui.element | None = None
        self.render()

    def add_input(self):
        with self.inputs_container:
            with ui.row().classes('items-center gap-2') as row:
                ui.textarea().props('outlined')
                ui.button(icon='remove', on_click=lambda: row.delete()).props('size=sm round')

    def collect_data(self) -> list[PromptTemplate]:
        prompt_templates = []
        for element in self.inputs_container.descendants():
            if isinstance(element, ui.textarea):
                prompt_templates.append(PromptTemplate.from_string(text=element.value))
        return prompt_templates

    def render(self):
        with self.card:
            with ui.card_section():
                ui.label('Prompt Template(s)').classes('text-h6')
            self.inputs_container = ui.element('div').classes('w-full')
            self.add_input()
            with ui.row().classes('w-full justify-center'):
                ui.button(icon='add', on_click=self.add_input)

            with ui.card_actions().classes('w-full justify-end'):
                ui.button(text='Cancel')
                ui.button(text='Save', on_click=self.collect_data)
