# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2018-04-29 01:57
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ontask', '0020_auto_20180429_1139'),
    ]

    operations = [
        migrations.AlterField(
            model_name='column',
            name='position',
            field=models.IntegerField(default=0, verbose_name='Column position (zero to insert last)'),
        ),
    ]