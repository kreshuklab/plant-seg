from magicgui import magicgui
from magicgui.widgets import Container

# from panseg.headless.headless import run_headless_workflow_from_path
from panseg.headless_gui.panseg_classic import widget_panseg_classic

all_workflows = {
    "PlantsegClassic": widget_panseg_classic,
}


@magicgui(
    auto_call=True,
    name={
        "label": "Mode",
        "tooltip": "Select the workflow to run",
        "choices": list(all_workflows.keys()),
    },
)
def workflow_selector(name: str = list(all_workflows.keys())[0]):
    for workflow in all_workflows.values():
        workflow.hide()

    all_workflows[name].show()


if __name__ == "__main__":
    gui_container = Container(
        widgets=[workflow_selector, *all_workflows.values()], labels=False
    )
    workflow_selector()
    gui_container.show(run=True)
