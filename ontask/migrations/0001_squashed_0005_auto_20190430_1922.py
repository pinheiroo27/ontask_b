# Generated by Django 2.2.1 on 2019-05-24 04:43

import django.contrib.postgres.fields.jsonb
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    replaces = [('ontask', '0001_table_initial'), ('ontask', '0002_auto_20180116_1510'), ('ontask', '0003_auto_20180116_2158'), ('ontask', '0004_auto_20180511_1528'), ('ontask', '0005_auto_20190430_1922')]

    initial = True

    dependencies = [
        ('ontask', '0013_auto_20171209_0809'),
    ]

    operations = [
        migrations.CreateModel(
            name='View',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('description_text', models.CharField(blank=True, default='', max_length=512)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
                ('formula', django.contrib.postgres.fields.jsonb.JSONField(blank=True, default=dict, help_text='Preselect rows satisfying this condition', null=True, verbose_name='Subset of rows to show')),
                ('columns', models.ManyToManyField(related_name='views', to='ontask.Column', verbose_name='Subset of columns to show')),
                ('workflow', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='views', to='ontask.Workflow')),
            ],
            options={
                'ordering': ['name'],
                'unique_together': {('name', 'workflow')},
            },
        ),
    ]