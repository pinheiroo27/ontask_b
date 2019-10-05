# -*- coding: utf-8 -*-

"""Abstract model for connections."""
from typing import Dict

from django.db import models
from django.utils.translation import ugettext_lazy as _

from ontask.models.const import CHAR_FIELD_LONG_SIZE
from ontask.models.logs import Log


class Connection(models.Model):
    """Model representing a connection to a data source.

    @DynamicAttrs
    """

    # Connection name
    name = models.CharField(
        verbose_name=_('Name'),
        max_length=CHAR_FIELD_LONG_SIZE,
        blank=False,
        unique=True)

    # Description
    description_text = models.CharField(
        verbose_name=_('Description'),
        max_length=CHAR_FIELD_LONG_SIZE,
        default='',
        blank=True)

    @classmethod
    def get(cls, pk):
        """Get the object with the given PK. Must be overwritten."""
        raise NotImplementedError

    def __str__(self):
        """Render with name field."""
        return self.name

    def get_display_dict(self) -> Dict:
        """Create dictionary with (verbose_name, value)"""
        return {
            self._meta.get_field(key.name).verbose_name.title():
                self.__dict__[key.name]
            for key in self._meta.get_fields()
            if key.name != 'id'}

    def log(self, user, operation_type: str, **kwargs) -> int:
        """Function to register an event."""
        payload = {}
        payload.upate(kwargs)
        return Log.objects.register(self, user, operation_type, None, payload)

    class Meta:
        """Define as abstract and the ordering criteria."""

        abstract = True
        ordering = ['name']
