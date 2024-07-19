# Create user if not already present
create_user_if_not_exists() {
    if ! id -u "${USER}" >/dev/null 2>&1; then
        useradd -u "${_UID}" -s /bin/bash -m "${USER}"
    fi
}

# Modify group ID if the group doesn't exist
modify_gid_if_not_exists() {
    if ! getent group "${_GID}" >/dev/null 2>&1; then
        groupmod -g "${_GID}" "${USER}"
    fi
}

# Add user to various groups
add_user_to_groups() {
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}"
    usermod -a -G "${USER}" "${USER}"
    usermod -a -G video "${USER}"
    adduser "${USER}" dialout
}

# Update permissions for user
update_user_permissions() {
    chown -R "${USER}:${USER}" "${PROJECT_WS}"
    chown -R "${USER}:${USER}" /root
    chmod 755 "${PROJECT_WS}"
}

create_user_if_not_exists
modify_gid_if_not_exists
add_user_to_groups
update_user_permissions

# Switch to user and start a new shell
su ${USER}
exec bin/bash