# Generated by Django 2.2.3 on 2019-08-02 08:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('action', '0002_auto_20190524_1639'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='action',
            table='action',
        ),
        migrations.AlterModelTable(
            name='actioncolumnconditiontuple',
            table='condition',
        ),
    ]
