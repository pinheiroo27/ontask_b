# Generated by Django 2.2.8 on 2019-12-21 23:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ontask', '0021_auto_20191221_1852'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='scheduledoperation',
            unique_together={('name', 'workflow')},
        ),
    ]
