# Generated by Django 2.2.8 on 2019-12-29 22:07

from django.db import migrations


def move_excluded_items_to_payload(apps, schema_editor):
    if schema_editor.connection.alias != 'default':
        return

    # Traverse the scheduled actions and move the exclude_items field from
    # the object field to the payload
    ScheduledOperation = apps.get_model('ontask', 'ScheduledOperation')
    for item in ScheduledOperation.objects.all():
        if item.exclude_values:
            item.payload['exclude_values'] = item.exclude_values
            item.save()


class Migration(migrations.Migration):
    """Move excluded_items to the paylaod."""

    dependencies = [
        ('ontask', '0022_auto_20191222_0931'),
    ]

    operations = [
        migrations.RunPython(move_excluded_items_to_payload),
    ]
