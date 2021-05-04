from datetime import datetime
import json

import click
from rich.traceback import install

from model import Model
from terminal import Progress, console, print, progress_layout, Confirm

install()

def list_deco(decorators):
    def decorator(f):
        for d in reversed(decorators):
            f = d(f)
        return f
    return decorator

PUBLIC_OPTIONS = [
    click.option('-d', '--dataset', default='SST5', help="Dataset used."),
    click.option('-e', '--epoch', default=10),
    click.option('-b', '--batch', default=500),
    click.option('-i', '--iteration', default=1),
    click.option('-f', '--feature_size', default=20000),
    click.option('-r', '--rand_lr/--no-rand_lr', default=False, help="Use random learning rate"),
    click.option('-l', '--lr', default=0.01, help="Learning rate or coefficient of random learning rate."),
    click.option('--trigram/--no-trigram', default=False),
    click.option('--stopword/--no-stopword', default=True),
    click.option('--lemmatize/--no-lemmatize', default=True),
    click.option('--penalty', default=0.0),
    click.option('--feature_pick', type=click.Choice(['top', 'freq']), default='top'),
    click.option('--feature_drop', default=0.0)
]

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        ctx.invoke(tour)

@cli.command()
@click.pass_context
def tour(ctx: click.Context):
    prompt_baseline_sst5 = Confirm.ask('[bold red]Baseline for SST5, typically [blue]<1 min[/blue]. Proceed?')
    if prompt_baseline_sst5: ctx.invoke(train, dataset='SST5', output='tour-baseline-SST5', export='tour-baseline-SST5')
    
    prompt_baseline_20news = Confirm.ask('[bold red]Baseline for 20news, typically [blue]8 mins[/blue]. Proceed?')
    if prompt_baseline_20news: ctx.invoke(train, dataset='20news', output='tour-baseline-20news', export='tour-baseline-20news')

    prompt_ablation = Confirm.ask('[bold red]Ablation for SST5 on learning rate, typically [blue]2.5 mins[/blue]. Proceed?')
    if prompt_ablation: ctx.invoke(ablation, dataset='SST5', param='lr', values=['0.01', '0.025', '0.05'])

    prompt_preset = Confirm.ask('[bold red]Preset for 20news, typically [blue]<1 min[/blue]. Proceed?')
    if prompt_preset: ctx.invoke(preset, filename='tour-preset')

    print('[bold red]The tour is over. Thank you!')


@cli.command()
@click.option('-n', '--name', default='', help='A name assigned for improving readability.')
@click.option('-o', '--output', default='', help='Output results to [filename].md under records/')
@click.option('-j', '--export', default='', help='Export model to [filename].json under presets/')
@list_deco(PUBLIC_OPTIONS)
def train(name, output, export, **kwargs):
    if kwargs['rand_lr']: kwargs['lr_rand_coef'] = kwargs['lr']
    else: kwargs['lr_fixed'] = kwargs['lr']
    del kwargs['lr']

    if name == '': name = datetime.now().isoformat()
    model = Model(**kwargs)

    print(f'\n[bold red]Train {name}')
    if output != '': print(f'[red]Results appended at records/{output}.md')
    if export != '': print(f'[red]Model exported at presets/{export}.json')
    print(model.hyper.__dict__)

    if output != '':
        with open(f'records/{output}.md', 'a') as file_output:
            file_output.write(f'\n## {name} on {datetime.now().strftime(r"%Y%m%d-%H%M%S")}\n\n')
            file_output.write('```json\n')
            json.dump(model.hyper.__dict__, file_output); file_output.write('\n```\n\n')
    
    max_accuracy = 0
    for epoch in range(model.hyper.epoch):
        print(f'[bold red]Epoch #{epoch+1} of {model.hyper.epoch}')
        model.df_train = model.df_train.sample(frac=1).reset_index(drop=True)
        model.train()
        if epoch == 0: model.prepare_test()
        model.test()
        if max_accuracy < model.accuracy: 
            max_accuracy = model.accuracy
            if export != '':
                dict_export = {
                    'hyper': model.hyper.__dict__,
                    'features': model.features.tolist(),
                    'weights': model.weights.tolist(),
                    'count_label': model.count_label,
                }
                with open(f'presets/{export}.json', 'w') as file_export:
                    json.dump(dict_export, file_export)

        print(f'[bold green]Epoch #{epoch+1}, training accuracy {model.train_accuracy*100:.9f}, testing accuracy [bold blue]{model.accuracy*100:.9f}[/bold blue] of [bold red]{max_accuracy*100:.9f}[/bold red] max\n')

        if output != '':
            with open(f'records/{output}.md', 'a') as file_output:
                file_output.write(f'epoch {epoch+1:2}, train_acc {model.train_accuracy*100:.9f}, acc _{model.accuracy*100:.9f}_, max **{max_accuracy*100:.9f}**\n')

    return model, max_accuracy

@cli.command()
@click.option('-p', '--param', help='The param ablated')
@click.option('-v', '--value', 'values', multiple=True, help='The values for the param')
@list_deco(PUBLIC_OPTIONS)
@click.pass_context
def ablation(ctx: click.Context, param, values, **kwargs):
    timestamp_ablation = datetime.now()
    for value in values:
        if value.isnumeric(): val = int(value)
        elif '.' in value: val = float(value)
        elif value in {'True', 'False'}: val = value == 'True'
        else: val = value
        kwargs[param] = val
        ctx.invoke(train, name=f'ablation-{param}-{value}', output=f'ablation-{kwargs["dataset"]}-{param}-{timestamp_ablation.strftime(r"%Y%m%d-%H%M%S")}', **kwargs)

@cli.command()
@click.argument('filename')
@click.pass_context
def preset(ctx, filename):
    with open(f'presets/{filename}.json', 'r') as file_input:
        dict_preset = json.load(file_input)

    model = Model(preset=dict_preset)
    print(f'\n[bold red]Preset {filename}')
    print(model.hyper.__dict__)
    model.prepare_test()
    model.test()
    print(f'[bold green]Testing accuracy [bold red]{model.accuracy*100:.9f}[/bold red]\n')

if __name__ == '__main__':
    cli()
    