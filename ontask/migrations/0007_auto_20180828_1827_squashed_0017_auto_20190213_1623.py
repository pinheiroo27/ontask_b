# Generated by Django 2.2.1 on 2019-05-24 02:18

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    replaces = [('ontask', '0007_auto_20180828_1827'), ('ontask', '0008_auto_20180902_1856'), ('ontask', '0009_auto_20180906_1802'), ('ontask', '0010_auto_20181018_0832'), ('ontask', '0011_auto_20181020_2208'), ('ontask', '0012_auto_20181124_1847'), ('ontask', '0013_auto_20181126_1854'), ('ontask', '0011_auto_20181020_2206'), ('ontask', '0012_merge_20181111_2142'), ('ontask', '0014_merge_20181201_1627'), ('ontask', '0015_auto_20181207_0539'), ('ontask', '0016_auto_20181219_2214'), ('ontask', '0017_auto_20190213_1623')]

    dependencies = [
        ('ontask', '0006_auto_20180825_1123'),
    ]

    operations = [
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), (('download_zip_action',), 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_canvas_email_execute', 'Execute scheduled canvas email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_canvas_email_sent', 'Canvas Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_canvas_email_execute', 'Execute scheduled canvas email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_canvas_email_sent', 'Canvas Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_canvas_email_execute', 'Execute scheduled canvas email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
        migrations.AlterField(
            model_name='log',
            name='workflow',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='logs', to='ontask.Workflow'),
        ),
        migrations.AlterField(
            model_name='log',
            name='name',
            field=models.CharField(choices=[('workflow_create', 'Workflow created'), ('workflow_update', 'Workflow updated'), ('workflow_delete', 'Workflow deleted'), ('workflow_data_upload', 'Data uploaded to workflow'), ('workflow_data_merge', 'Data merged into workflow'), ('workflow_data_failedmerge', 'Failed data merge into workflow'), ('workflow_data_flush', 'Workflow data flushed'), ('workflow_attribute_create', 'New attribute in workflow'), ('workflow_attribute_update', 'Attributes updated in workflow'), ('workflow_attribute_delete', 'Attribute deleted'), ('workflow_share_add', 'User share added'), ('workflow_share_delete', 'User share deleted'), ('workflow_import', 'Import workflow'), ('workflow_clone', 'Workflow cloned'), ('column_add', 'Column added'), ('column_add_formula', 'Column with formula created'), ('column_add_random', 'Column with random values created'), ('column_rename', 'Column renamed'), ('column_delete', 'Column deleted'), ('column_clone', 'Column cloned'), ('column_restrict', 'Column restricted'), ('action_create', 'Action created'), ('action_update', 'Action updated'), ('action_delete', 'Action deleted'), ('action_clone', 'Action cloned'), ('action_email_sent', 'Emails sent'), ('action_canvas_email_sent', 'Canvas Emails sent'), ('action_email_notify', 'Notification email sent'), ('action_email_read', 'Email read'), ('action_serve_toggled', 'Action URL toggled'), ('action_served_execute', 'Action served'), ('action_import', 'Action imported'), ('action_json_sent', 'Emails sent'), ('condition_create', 'Condition created'), ('condition_update', 'Condition updated'), ('condition_delete', 'Condition deleted'), ('condition_clone', 'Condition cloned'), ('tablerow_update', 'Table row updated'), ('tablerow_create', 'Table row created'), ('view_create', 'Table view created'), ('view_edit', 'Table view edited'), ('view_delete', 'Table view deleted'), ('view_clone', 'Table view cloned'), ('filter_create', 'Filter created'), ('filter_update', 'Filter updated'), ('filter_delete', 'Filter deleted'), ('plugin_create', 'Plugin created'), ('plugin_update', 'Plugin updated'), ('plugin_delete', 'Plugin deleted'), ('plugin_execute', 'Plugin executed'), ('sql_connection_create', 'SQL connection created'), ('sql_connection_edit', 'SQL connection updated'), ('sql_connection_delete', 'SQL connection deleted'), ('sql_connection_clone', 'SQL connection cloned'), ('schedule_email_edit', 'Edit scheduled email action'), ('schedule_email_delete', 'Delete scheduled email action'), ('schedule_email_execute', 'Execute scheduled email action'), ('schedule_canvas_email_execute', 'Execute scheduled canvas email action'), ('schedule_canvas_email_delete', 'Delete scheduled canvas email action'), ('download_zip_action', 'Download a ZIP with personalized text'), ('schedule_json_edit', 'Edit scheduled JSON action'), ('schedule_json_delete', 'Delete scheduled JSON action'), ('schedule_json_execute', 'Execute scheduled JSON action')], max_length=256),
        ),
    ]
