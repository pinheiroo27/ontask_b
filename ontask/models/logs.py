# -*- coding: utf-8 -*-

"""Model for OnTask Logs."""

import json

from django.conf import settings
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.utils.functional import cached_property
from django.utils.translation import ugettext_lazy as _

FIELD_NAME_LENGH = 256


class LogManager(models.Manager):
    """Manager to create elements with the right parameters."""

    def register(self, user, name, workflow, payload):
        """Handle user, name, workflow and payload."""
        log_item = self.create(
            user=user,
            name=name,
            workflow=workflow,
            payload=payload)
        return log_item


class Log(models.Model):
    """Model to encode logs in OnTask.

    @DynamicAttrs
    """

    ACTION_CANVAS_EMAIL_SENT = 'action_canvas_email_sent'
    ACTION_CLONE = 'action_clone'
    ACTION_CREATE = 'action_create'
    ACTION_DELETE = 'action_delete'
    ACTION_DOWNLOAD = 'download_zip_action'
    ACTION_EMAIL_NOTIFY = 'action_email_notify'
    ACTION_EMAIL_READ = 'action_email_read'
    ACTION_EMAIL_SENT = 'action_email_sent'
    ACTION_IMPORT = 'action_import'
    ACTION_JSON_SENT = 'action_json_sent'
    ACTION_LIST_EMAIL_SENT = 'action_list_email_sent'
    ACTION_QUESTION_ADD = 'question_add'
    ACTION_QUESTION_TOGGLE_CHANGES = 'question_toggle_changes'
    ACTION_RUBRIC_CRITERION_ADD = 'action_rubric_criterion_add'
    ACTION_RUBRIC_CRITERION_EDIT = 'action_rubric_criterion_edit'
    ACTION_RUBRIC_CRITERION_DELETE = 'action_rubric_criterion_delete'
    ACTION_RUBRIC_CELL_EDIT = 'action_rubriccell_edit'
    ACTION_RUBRIC_LOA_EDIT = 'action_rubric_loa_edit'
    ACTION_RUN_CANVAS_EMAIL = 'action_run_canvas_email'
    ACTION_RUN_EMAIL = 'schedule_email_execute'
    ACTION_RUN_JSON = 'schedule_json_execute'
    ACTION_RUN_JSON_LIST = 'schedule_send_list_execute'
    ACTION_RUN_SEND_LIST = 'schedule_send_list_execute'
    ACTION_SERVE_TOGGLED = 'action_serve_toggled'
    ACTION_SERVED_EXECUTE = 'action_served_execute'
    ACTION_SURVEY_INPUT = 'survey_input'
    ACTION_UPDATE = 'action_update'
    ATHENA_CONNECTION_CREATE = 'athena_connection_create'
    ATHENA_CONNECTION_EDIT = 'athena_connection_edit'
    ATHENA_CONNECTION_DELETE = 'athena_connection_delete'
    ATHENA_CONNECTION_CLONE = 'athena_connection_clone'
    ATHENA_CONNECTION_TOGGLE = 'athena_connection_toggle'
    COLUMN_ADD = 'column_add'
    COLUMN_ADD_FORMULA = 'column_add_formula'
    COLUMN_ADD_RANDOM = 'column_add_random'
    COLUMN_CLONE = 'column_clone'
    COLUMN_DELETE = 'column_delete'
    COLUMN_EDIT = 'column_edit'
    COLUMN_RESTRICT = 'column_restrict'
    CONDITION_CLONE = 'condition_clone'
    CONDITION_CREATE = 'condition_create'
    CONDITION_DELETE = 'condition_delete'
    CONDITION_UPDATE = 'condition_update'
    PLUGIN_CREATE = 'plugin_create'
    PLUGIN_DELETE = 'plugin_delete'
    PLUGIN_EXECUTE = 'plugin_execute'
    PLUGIN_UPDATE = 'plugin_update'
    SCHEDULE_EMAIL_EDIT = 'schedule_email_edit'
    SCHEDULE_EMAIL_DELETE = 'schedule_email_delete'
    SCHEDULE_JSON_LIST_EDIT = 'schedule_send_list_edit'
    SCHEDULE_JSON_LIST_DELETE = 'schedule_send_list_delete'
    SCHEDULE_SEND_LIST_EDIT = 'schedule_send_list_edit'
    SCHEDULE_SEND_LIST_DELETE = 'schedule_send_list_delete'
    SCHEDULE_CANVAS_EMAIL_EDIT = 'schedule_canvas_email_edit'
    SCHEDULE_CANVAS_EMAIL_DELETE = 'schedule_canvas_email_delete'
    SCHEDULE_JSON_EDIT = 'schedule_json_edit'
    SCHEDULE_JSON_DELETE = 'schedule_json_delete'
    SQL_CONNECTION_CLONE = 'sql_connection_clone'
    SQL_CONNECTION_CREATE = 'sql_connection_create'
    SQL_CONNECTION_DELETE = 'sql_connection_delete'
    SQL_CONNECTION_EDIT = 'sql_connection_edit'
    SQL_CONNECTION_TOGGLE = 'sql_connection_toggle'
    VIEW_CLONE = 'view_clone'
    VIEW_CREATE = 'view_create'
    VIEW_DELETE = 'view_delete'
    VIEW_EDIT = 'view_edit'
    WORKFLOW_ATTRIBUTE_CREATE = 'workflow_attribute_create'
    WORKFLOW_ATTRIBUTE_UPDATE = 'workflow_attribute_update'
    WORKFLOW_ATTRIBUTE_DELETE = 'workflow_attribute_delete'
    WORKFLOW_CLONE = 'workflow_clone'
    WORKFLOW_CREATE = 'workflow_create'
    WORKFLOW_DATA_FAILEDMERGE = 'workflow_data_failedmerge'
    WORKFLOW_DATA_FLUSH = 'workflow_data_flush'
    WORKFLOW_DATA_MERGE = 'workflow_data_merge'
    WORKFLOW_DATA_ROW_UPDATE = 'tablerow_update'
    WORKFLOW_DATA_ROW_CREATE = 'tablerow_create'
    WORKFLOW_DATA_UPLOAD = 'workflow_data_upload'
    WORKFLOW_DELETE = 'workflow_delete'
    WORKFLOW_IMPORT = 'workflow_import'
    WORKFLOW_SHARE_ADD = 'workflow_share_add'
    WORKFLOW_SHARE_DELETE = 'workflow_share_delete'
    WORKFLOW_STAR = 'workflow_star'
    WORKFLOW_UPDATE = 'workflow_update'
    WORKFLOW_UPDATE_LUSERS = 'workflow_update_lusers'

    LOG_TYPES = [
        (ACTION_CANVAS_EMAIL_SENT, _('Canvas Emails sent')),
        (ACTION_CLONE, _('Action cloned')),
        (ACTION_CREATE, _('Action created')),
        (ACTION_DELETE, _('Action deleted')),
        (ACTION_DOWNLOAD, _('Download a ZIP with one file per text')),
        (ACTION_EMAIL_NOTIFY, _('Notification email sent')),
        (ACTION_EMAIL_READ, _('Email read')),
        (ACTION_EMAIL_SENT, _('Emails sent')),
        (ACTION_IMPORT, _('Action imported')),
        (ACTION_JSON_SENT, _('Emails sent')),
        (ACTION_LIST_EMAIL_SENT, _('Email with data list sent')),
        (ACTION_QUESTION_ADD, _('Question added')),
        (ACTION_QUESTION_TOGGLE_CHANGES, _('Question toggle changes')),
        (ACTION_RUBRIC_CRITERION_ADD, _('Add a rubric criterion')),
        (ACTION_RUBRIC_CRITERION_EDIT, _('Edit rubric criterion')),
        (ACTION_RUBRIC_CRITERION_DELETE, _('Delete rubric criterion')),
        (ACTION_RUBRIC_CELL_EDIT, _('Rubric cell edit')),
        (ACTION_RUBRIC_LOA_EDIT, _('Rubric level of attainment edit')),
        (ACTION_RUN_CANVAS_EMAIL, _('Execute scheduled canvas email action')),
        (ACTION_RUN_EMAIL, _('Execute scheduled email action')),
        (ACTION_RUN_JSON, _('Execute scheduled JSON action')),
        (ACTION_RUN_JSON_LIST, _('Execute scheduled JSON list action')),
        (ACTION_RUN_SEND_LIST, _('Execute scheduled send list action')),
        (ACTION_SERVE_TOGGLED, _('Action URL toggled')),
        (ACTION_SERVED_EXECUTE, _('Action served')),
        (ACTION_SURVEY_INPUT, _('Survey data input')),
        (ACTION_UPDATE, _('Action updated')),
        (ATHENA_CONNECTION_CLONE, _('Athena connection cloned')),
        (ATHENA_CONNECTION_CREATE, _('Athena connection created')),
        (ATHENA_CONNECTION_DELETE, _('Athena connection deleted')),
        (ATHENA_CONNECTION_EDIT, _('Athena connection updated')),
        (ATHENA_CONNECTION_TOGGLE, _('SQL connection toggled')),
        (COLUMN_ADD, _('Column added')),
        (COLUMN_ADD_FORMULA, _('Column with formula created')),
        (COLUMN_ADD_RANDOM, _('Column with random values created')),
        (COLUMN_CLONE, _('Column cloned')),
        (COLUMN_DELETE, _('Column deleted')),
        (COLUMN_EDIT, _('Column edited')),
        (COLUMN_RESTRICT, _('Column restricted')),
        (CONDITION_CLONE, _('Condition cloned')),
        (CONDITION_CREATE, _('Condition created')),
        (CONDITION_DELETE, _('Condition deleted')),
        (CONDITION_UPDATE, _('Condition updated')),
        (PLUGIN_CREATE, _('Plugin created')),
        (PLUGIN_DELETE, _('Plugin deleted')),
        (PLUGIN_EXECUTE, _('Plugin executed')),
        (PLUGIN_UPDATE, _('Plugin updated')),
        (SCHEDULE_CANVAS_EMAIL_EDIT,
         _('Edit scheduled canvas email action')),
        (SCHEDULE_CANVAS_EMAIL_DELETE,
         _('Delete scheduled canvas email action')),
        (SCHEDULE_EMAIL_EDIT, _('Edit scheduled email action')),
        (SCHEDULE_EMAIL_DELETE, _('Delete scheduled email action')),
        (SCHEDULE_JSON_EDIT, _('Edit scheduled JSON action')),
        (SCHEDULE_JSON_DELETE, _('Delete scheduled JSON action')),
        (SCHEDULE_JSON_LIST_EDIT, _('Edit scheduled JSON list action')),
        (SCHEDULE_JSON_LIST_DELETE, _('Delete scheduled JSON list action')),
        (SCHEDULE_SEND_LIST_EDIT, _('Edit scheduled send list action')),
        (SCHEDULE_SEND_LIST_DELETE, _('Delete scheduled send list action')),
        (SQL_CONNECTION_CLONE, _('SQL connection cloned')),
        (SQL_CONNECTION_CREATE, _('SQL connection created')),
        (SQL_CONNECTION_DELETE, _('SQL connection deleted')),
        (SQL_CONNECTION_EDIT, _('SQL connection updated')),
        (SQL_CONNECTION_TOGGLE, _('SQL connection toggled')),
        (VIEW_CREATE, _('Table view created')),
        (VIEW_EDIT, _('Table view edited')),
        (VIEW_DELETE, _('Table view deleted')),
        (VIEW_CLONE, _('Table view cloned')),
        (WORKFLOW_ATTRIBUTE_CREATE, _('New attribute in workflow')),
        (WORKFLOW_ATTRIBUTE_UPDATE, _('Attributes updated in workflow')),
        (WORKFLOW_ATTRIBUTE_DELETE, _('Attribute deleted')),
        (WORKFLOW_CLONE, _('Workflow cloned')),
        (WORKFLOW_CREATE, _('Workflow created')),
        (WORKFLOW_DATA_FAILEDMERGE, _('Failed data merge into workflow')),
        (WORKFLOW_DATA_FLUSH, _('Workflow data flushed')),
        (WORKFLOW_DATA_MERGE, _('Data merged into workflow')),
        (WORKFLOW_DATA_ROW_CREATE, _('Table row created')),
        (WORKFLOW_DATA_ROW_UPDATE, _('Table row updated')),
        (WORKFLOW_DATA_UPLOAD, _('Data uploaded to workflow')),
        (WORKFLOW_DELETE, _('Workflow deleted')),
        (WORKFLOW_IMPORT, _('Import workflow')),
        (WORKFLOW_SHARE_ADD, _('User share added')),
        (WORKFLOW_SHARE_DELETE, _('User share deleted')),
        (WORKFLOW_STAR, _('Toggle workflow star')),
        (WORKFLOW_UPDATE, _('Workflow updated')),
        (WORKFLOW_UPDATE_LUSERS, _('Update list of workflow users')),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        db_index=True,
        on_delete=models.CASCADE,
        null=False,
        blank=False)

    created = models.DateTimeField(auto_now_add=True, null=False, blank=False)

    modified = models.DateTimeField(auto_now=True, null=False)

    # Type of event logged see above
    name = models.CharField(
        max_length=FIELD_NAME_LENGH,
        blank=False,
        choices=LOG_TYPES)

    workflow = models.ForeignKey(
        'Workflow',
        db_index=True,
        on_delete=models.CASCADE,
        null=True,
        related_name='logs')

    # JSON element with additional information
    payload = JSONField(
        default=dict,
        blank=True,
        null=True,
        verbose_name=_('payload'))

    # Use our own manager
    objects = LogManager()  # noqa: Z110

    def get_payload(self):
        """Access the payload information.

        If using a DB that supports JSON this function should be rewritten (
        to be transparent).

        :return: The JSON structure with the payload
        """
        if self.payload == '':
            return {}

        return json.loads(self.payload)

    def set_payload(self, payload):
        """Save the payload structure as text.

        If using a DB that supports JSON, this function should be rewritten.

        :return: Nothing.
        """
        self.payload = json.dumps(payload)

    def __unicode__(self):
        """Represent as a tuple."""
        return '%s %s %s %s' % (
            self.user,
            self.created,
            self.name,
            self.payload)

    @cached_property
    def log_useremail(self):
        """Return the user email."""
        return self.user.email
