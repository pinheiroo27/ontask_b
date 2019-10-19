# -*- coding: utf-8 -*-
# Generated by Django 1.11.14 on 2018-08-28 08:57
from __future__ import unicode_literals

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ontask', '0028_auto_20180827_2121'),
    ]

    operations = [
        migrations.AlterField(
            model_name='scheduledaction',
            name='item_column',
            field=models.ForeignKey(blank=True, db_index=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='scheduled_actions', to='ontask.Column', verbose_name='Column to select the elements for the action'),
        ),
    ]