# Generated by Django 2.2.5 on 2019-10-07 10:48

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('ontask', '0046_auto_20191007_2057'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='ScheduledAction',
            new_name='ScheduledOperation',
        ),
    ]