# Generated by Django 2.1.7 on 2019-03-22 14:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ontask', '0054_auto_20190318_2144'),
    ]

    operations = [
        migrations.AddField(
            model_name='action',
            name='nrows_all_false',
            field=models.IntegerField(blank=True, default=-1, null=True, verbose_name='Number of rows with all conditions false'),
        ),
    ]